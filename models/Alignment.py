import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn

from models.CtrlHair.shape_branch.config import cfg as cfg_mask
from models.CtrlHair.shape_branch.solver import get_hair_face_code, get_new_shape, Solver as SolverMask
from models.Encoders import RotateModel
from models.Net import Net, get_segmentation
from models.sean_codes.models.pix2pix_model import Pix2PixModel, SEAN_OPT, encode_sean, decode_sean
from utils.image_utils import DilateErosion
from utils.save_utils import save_vis_mask, save_gen_image, save_latents


class Alignment(nn.Module):
    """
    Module for transferring the desired hair shape
    """

    def __init__(self, opts, latent_encoder=None, net=None):
        super().__init__()
        self.opts = opts
        self.latent_encoder = latent_encoder
        if not net:
            self.net = Net(self.opts)
        else:
            self.net = net

        self.sean_model = Pix2PixModel(SEAN_OPT)
        self.sean_model.eval()

        solver_mask = SolverMask(cfg_mask, device=self.opts.device, local_rank=-1, training=False)
        self.mask_generator = solver_mask.gen
        self.mask_generator.load_state_dict(torch.load('pretrained_models/ShapeAdaptor/mask_generator.pth'))

        self.rotate_model = RotateModel()
        self.rotate_model.load_state_dict(torch.load(self.opts.rotate_checkpoint)['model_state_dict'])
        self.rotate_model.to(self.opts.device).eval()

        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.opts.device)
        self.to_bisenet = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    @torch.inference_mode()
    def shape_module(self, im_name1: str, im_name2: str, name_to_embed, only_target=True, **kwargs):
        device = self.opts.device

        # load images
        img1_in = name_to_embed[im_name1]['image_256']
        img2_in = name_to_embed[im_name2]['image_256']

        # load latents
        latent_W_1 = name_to_embed[im_name1]["W"]
        latent_W_2 = name_to_embed[im_name2]["W"]

        # load masks
        inp_mask1 = name_to_embed[im_name1]['mask']
        inp_mask2 = name_to_embed[im_name2]['mask']

        # Rotate stage
        if img1_in is not img2_in:
            rotate_to = self.rotate_model(latent_W_2[:, :6], latent_W_1[:, :6])
            rotate_to = torch.cat((rotate_to, latent_W_2[:, 6:]), dim=1)
            I_rot, _ = self.net.generator([rotate_to], input_is_latent=True, return_latents=False)

            I_rot_to_seg = ((I_rot + 1) / 2).clip(0, 1)
            I_rot_to_seg = self.to_bisenet(I_rot_to_seg)
            rot_mask = get_segmentation(I_rot_to_seg)
        else:
            I_rot = None
            rot_mask = inp_mask2

        # Shape Adaptor
        if img1_in is not img2_in:
            face_1, hair_1 = get_hair_face_code(self.mask_generator, inp_mask1[0, 0, ...])
            face_2, hair_2 = get_hair_face_code(self.mask_generator, rot_mask[0, 0, ...])

            target_mask = get_new_shape(self.mask_generator, face_1, hair_2)[None, None]
        else:
            target_mask = inp_mask1

        # Hair mask
        hair_mask_target = torch.where(target_mask == 13, torch.ones_like(target_mask, device=device),
                                       torch.zeros_like(target_mask, device=device))

        if self.opts.save_all:
            exp_name = exp_name if (exp_name := kwargs.get('exp_name')) is not None else ""
            output_dir = self.opts.save_all_dir / exp_name
            if I_rot is not None:
                save_gen_image(output_dir, 'Shape', f'{im_name2}_rotate_to_{im_name1}.png', I_rot)
            save_vis_mask(output_dir, 'Shape', f'mask_{im_name1}.png', inp_mask1)
            save_vis_mask(output_dir, 'Shape', f'mask_{im_name2}.png', inp_mask2)
            save_vis_mask(output_dir, 'Shape', f'mask_{im_name2}_rotate_to_{im_name1}.png', rot_mask)
            save_vis_mask(output_dir, 'Shape', f'mask_{im_name1}_{im_name2}_target.png', target_mask)

        if only_target:
            return {'HM_X': hair_mask_target}
        else:
            hair_mask1 = torch.where(inp_mask1 == 13, torch.ones_like(inp_mask1, device=device),
                                     torch.zeros_like(inp_mask1, device=device))
            hair_mask2 = torch.where(inp_mask2 == 13, torch.ones_like(inp_mask2, device=device),
                                     torch.zeros_like(inp_mask2, device=device))

            return inp_mask1, hair_mask1, inp_mask2, hair_mask2, target_mask, hair_mask_target

    @torch.inference_mode()
    def align_images(self, im_name1, im_name2, name_to_embed, **kwargs):
        # load images
        img1_in = name_to_embed[im_name1]['image_256']
        img2_in = name_to_embed[im_name2]['image_256']

        # load latents
        latent_S_1, latent_F_1 = name_to_embed[im_name1]["S"], name_to_embed[im_name1]["F"]
        latent_S_2, latent_F_2 = name_to_embed[im_name2]["S"], name_to_embed[im_name2]["F"]

        # Shape Module
        if img1_in is img2_in:
            hair_mask_target = self.shape_module(im_name1, im_name2, name_to_embed, only_target=True, **kwargs)['HM_X']
            return {'latent_F_align': latent_F_1, 'HM_X': hair_mask_target}

        inp_mask1, hair_mask1, inp_mask2, hair_mask2, target_mask, hair_mask_target = (
            self.shape_module(im_name1, im_name2, name_to_embed, only_target=False, **kwargs)
        )

        images = torch.cat([img1_in, img2_in], dim=0)
        labels = torch.cat([inp_mask1, inp_mask2], dim=0)

        # SEAN for inpaint
        img1_code, img2_code = encode_sean(self.sean_model, images, labels)

        gen1_sean = decode_sean(self.sean_model, img1_code.unsqueeze(0), target_mask)
        gen2_sean = decode_sean(self.sean_model, img2_code.unsqueeze(0), target_mask)

        # Encoding result in F from E4E
        enc_imgs = self.latent_encoder([gen1_sean, gen2_sean])
        intermediate_align, latent_inter = enc_imgs["F"][0].unsqueeze(0), enc_imgs["W"][0].unsqueeze(0)
        latent_F_out_new, latent_out = enc_imgs["F"][1].unsqueeze(0), enc_imgs["W"][1].unsqueeze(0)

        # Alignment of F space
        masks = [
            1 - (1 - hair_mask1) * (1 - hair_mask_target),
            hair_mask_target,
            hair_mask2 * hair_mask_target
        ]
        masks = torch.cat(masks, dim=0)
        # masks = T.functional.resize(masks, (1024, 1024), interpolation=T.InterpolationMode.NEAREST)

        dilate, erosion = self.dilate_erosion.mask(masks)
        free_mask = [
            dilate[0],
            erosion[1],
            erosion[2]
        ]
        free_mask = torch.stack(free_mask, dim=0)
        free_mask_down_32 = F.interpolate(free_mask.float(), size=(32, 32), mode='bicubic')
        interpolation_low = 1 - free_mask_down_32

        latent_F_align = intermediate_align + interpolation_low[0] * (latent_F_1 - intermediate_align)
        latent_F_align = latent_F_out_new + interpolation_low[1] * (latent_F_align - latent_F_out_new)
        latent_F_align = latent_F_2 + interpolation_low[2] * (latent_F_align - latent_F_2)

        if self.opts.save_all:
            exp_name = exp_name if (exp_name := kwargs.get('exp_name')) is not None else ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, 'Align', f'{im_name1}_{im_name2}_SEAN.png', gen1_sean)
            save_gen_image(output_dir, 'Align', f'{im_name2}_{im_name1}_SEAN.png', gen2_sean)

            img1_e4e = self.net.generator([latent_inter], input_is_latent=True, return_latents=False, start_layer=4,
                                          end_layer=8, layer_in=intermediate_align)[0]
            img2_e4e = self.net.generator([latent_out], input_is_latent=True, return_latents=False, start_layer=4,
                                          end_layer=8, layer_in=latent_F_out_new)[0]

            save_gen_image(output_dir, 'Align', f'{im_name1}_{im_name2}_e4e.png', img1_e4e)
            save_gen_image(output_dir, 'Align', f'{im_name2}_{im_name1}_e4e.png', img2_e4e)

            gen_im, _ = self.net.generator([latent_S_1], input_is_latent=True, return_latents=False, start_layer=4,
                                           end_layer=8, layer_in=latent_F_align)

            save_gen_image(output_dir, 'Align', f'{im_name1}_{im_name2}_output.png', gen_im)
            save_latents(output_dir, 'Align', f'{im_name1}_{im_name2}_F.npz', latent_F_align=latent_F_align)

        return {'latent_F_align': latent_F_align, 'HM_X': hair_mask_target}
