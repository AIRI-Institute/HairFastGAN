# Copyright (c) 2021, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import cv2
import glob
import numpy as np
import os
import face_alignment
import torch

from PIL import Image, ImageFilter
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from skimage import io
from torchvision import transforms, utils


def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image) 
    return open_cv_image[:, :, ::-1].copy() 


def cv2_to_pil(open_cv_image):
    return Image.fromarray(open_cv_image[:, :, ::-1].copy())


def put_text(img, text):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,50)
    fontScale              = 1.5
    fontColor              = (255,255,0)
    lineType               = 2
    return cv2.putText(img, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)


# Compare frames in two directory
def compare_frames(save_dir, origin_dir, target_dir, strs='Original,Projected,Manipulated', dim=None):
    
    os.makedirs(save_dir, exist_ok=True)
    try:
        if not isinstance(target_dir, list):
            target_dir = [target_dir]
        image_list = glob.glob1(origin_dir,'frame*')
        image_list.sort()
        for name in image_list:
            img_l = []
            for idx, dir_path in enumerate([origin_dir] + list(target_dir)):
                img_1 = cv2.imread(dir_path  + name)
                img_1 = put_text(img_1, strs.split(',')[idx])
                img_l.append(img_1)
            img = np.concatenate(img_l, axis=1)
            cv2.imwrite(save_dir + name, img)
    except FileNotFoundError:
        pass


# Save frames into video
def create_video(image_folder, fps=24, video_format='.mp4', resize_ratio=1):
    
    video_name = os.path.dirname(image_folder) + video_format
    img_list = glob.glob1(image_folder,'frame*')
    img_list.sort()
    frame = cv2.imread(os.path.join(image_folder, img_list[0]))
    frame = cv2.resize(frame, (0,0), fx=resize_ratio, fy=resize_ratio) 
    height, width, layers = frame.shape
    if video_format == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif video_format == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))
    for image_name in img_list:
        frame = cv2.imread(os.path.join(image_folder, image_name))
        frame = cv2.resize(frame, (0,0), fx=resize_ratio, fy=resize_ratio) 
        video.write(frame)


# Split video into frames
def video_to_frames(video_path, frame_path, img_format='.jpg', count_num=1000, resize=False):
    
    os.makedirs(frame_path, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        if resize:
            image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        cv2.imwrite(frame_path + '/frame%04d' % count + img_format, image)     
        success,image = vidcap.read()
        count += 1
        if count >= count_num:
            break

# Align faces
def align_frames(img_dir, save_dir, output_size=1024, transform_size=1024, optical_flow=True, gaussian=True, filter_size=3):
    
    os.makedirs(save_dir, exist_ok=True)

    # load face landmark detector
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

    # list images in the directory
    img_list = glob.glob1(img_dir, 'frame*')
    img_list.sort()

    # save align statistics
    stat_dict = {'quad':[], 'qsize':[], 'coord':[], 'crop':[]}
    lms = []
    for idx, img_name in enumerate(img_list):

        img_path = os.path.join(img_dir, img_name)
        img = io.imread(img_path)
        lm = []

        preds = fa.get_landmarks(img)
        for kk in range(68):
            lm.append((preds[0][kk][0], preds[0][kk][1]))
        
        # Eye distance
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_to_eye   = eye_right - eye_left
            
        if optical_flow:
            if idx > 0:
                s = int(np.hypot(*eye_to_eye)/4)
                lk_params = dict(winSize=(s, s), maxLevel=5, criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03))
                points_arr = np.array(lm, np.float32)
                points_prevarr = np.array(prev_lm, np.float32)
                points_arr,status, err = cv2.calcOpticalFlowPyrLK(prev_img, img, points_prevarr, points_arr, **lk_params)
                sigma =100
                points_arr_float = np.array(points_arr,np.float32)
                points = points_arr_float.tolist()
                for k in range(0, len(lm)):
                    d = cv2.norm(np.array(prev_lm[k]) - np.array(lm[k]))
                    alpha = np.exp(-d*d/sigma)
                    lm[k] = (1 - alpha) * np.array(lm[k]) + alpha * np.array(points[k])
            prev_img = img
            prev_lm = lm

        lms.append(lm)
    
    # Apply gaussian filter on landmarks
    if gaussian:
        lm_filtered = np.array(lms)
        for kk in range(68):
            lm_filtered[:, kk, 0] = gaussian_filter1d(lm_filtered[:, kk, 0], filter_size)
            lm_filtered[:, kk, 1] = gaussian_filter1d(lm_filtered[:, kk, 1], filter_size)
        lms = lm_filtered.tolist()

    # save landmarks
    landmark_out_dir = os.path.dirname(img_dir) + '_landmark/'
    os.makedirs(landmark_out_dir, exist_ok=True)

    for idx, img_name in enumerate(img_list):

        img_path = os.path.join(img_dir, img_name)
        img = io.imread(img_path)

        lm = lms[idx]
        img_lm = img.copy()
        for kk in range(68):
            img_lm = cv2.circle(img_lm, (int(lm[kk][0]),int(lm[kk][1])), radius=3, color=(255, 0, 255), thickness=-1)
        # Save landmark images
        cv2.imwrite(landmark_out_dir + img_name, img_lm[:,:,::-1])

        # Save mask images
        """
        seg_mask = np.zeros(img.shape, img.dtype)
        poly = np.array(lm[0:17] + lm[17:27][::-1], np.int32)
        cv2.fillPoly(seg_mask, [poly], (255, 255, 255))
        cv2.imwrite(img_dir + "mask%04d.jpg"%idx, seg_mask);
        """

        # Parse landmarks. 
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean([lm_eye_left[0], lm_eye_left[3]], axis=0)
        eye_right    = np.mean([lm_eye_right[0], lm_eye_right[3]], axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = np.array(lm_mouth_outer[0])
        mouth_right  = np.array(lm_mouth_outer[6])
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        stat_dict['coord'].append(quad)
        stat_dict['qsize'].append(qsize)

    # Apply gaussian filter on crops
    if gaussian:
        quads = np.array(stat_dict['coord'])
        quads = gaussian_filter1d(quads, 2*filter_size, axis=0)
        stat_dict['coord'] = quads.tolist()
        qsize = np.array(stat_dict['qsize'])
        qsize = gaussian_filter1d(qsize, 2*filter_size, axis=0)
        stat_dict['qsize'] = qsize.tolist()

    for idx, img_name in enumerate(img_list):

        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        
        qsize = stat_dict['qsize'][idx]
        quad = np.array(stat_dict['coord'][idx])

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]
        
        stat_dict['crop'].append(crop)
        stat_dict['quad'].append((quad + 0.5).flatten())

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]
        # Transform.
        img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)

        # resizing
        img_pil = img.resize((output_size, output_size), Image.LANCZOS)
        img_pil.save(save_dir+img_name)

    create_video(landmark_out_dir)
    np.save(save_dir+'stat_dict.npy', stat_dict)


def find_coeffs(pa, pb):

    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

# reproject aligned frames to the original video
def video_reproject(orig_dir_path, recon_dir_path, save_dir_path, state_dir_path, mask_dir_path, seamless=False):

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    img_list_0 = glob.glob1(orig_dir_path,'frame*')
    img_list_2 = glob.glob1(recon_dir_path,'frame*')
    img_list_0.sort()
    img_list_2.sort()
    stat_dict = np.load(state_dir_path + 'stat_dict.npy', allow_pickle=True).item()
    counter = len(img_list_2)

    for idx in range(counter):

        img_0 = Image.open(orig_dir_path + img_list_0[idx])
        img_2 = Image.open(recon_dir_path + img_list_2[idx])

        quad_f = stat_dict['quad'][idx]
        quad_0 = stat_dict['crop'][idx]

        coeffs = find_coeffs(
            [(quad_f[0], quad_f[1]), (quad_f[2] , quad_f[3]), (quad_f[4], quad_f[5]), (quad_f[6], quad_f[7])],
            [(0, 0), (0, 1024), (1024, 1024), (1024, 0)])
        crop_size = (quad_0[2] - quad_0[0], quad_0[3] - quad_0[1])
        img_2 = img_2.transform(crop_size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
        output = img_0.copy()
        output.paste(img_2, (int(quad_0[0]), int(quad_0[1])))
        
        """
        mask = cv2.imread(orig_dir_path + 'mask%04d.jpg'%idx)
        kernel = np.ones((10,10), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=5)
        """
        crop_mask = Image.open(mask_dir_path + img_list_0[idx])
        crop_mask = crop_mask.transform(crop_size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
        mask = Image.fromarray(np.zeros(np.array(img_0).shape, np.array(img_0).dtype))
        mask.paste(crop_mask, (int(quad_0[0]), int(quad_0[1])))
        mask = pil_to_cv2(mask)
        # Apply mask
        if not seamless:
            mask = cv2_to_pil(mask).filter(ImageFilter.GaussianBlur(radius=10)).convert('L')
            mask = np.array(mask)[:, :, np.newaxis]/255.
            output = np.array(img_0)*(1-mask) + np.array(output)*mask
            output = Image.fromarray(output.astype(np.uint8))
            output.save(save_dir_path + img_list_2[idx])
        else:
            src = pil_to_cv2(output)
            dst = pil_to_cv2(img_0)
            # clone
            br = cv2.boundingRect(cv2.split(mask)[0]) # bounding rect (x,y,width,height)
            center = (br[0] + br[2] // 2, br[1] + br[3] // 2)
            output = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
            cv2.imwrite(save_dir_path + img_list_2[idx], output)




# Align faces
def align_image(img_dir, save_dir, output_size=1024, transform_size=1024, format='*.png'):
    os.makedirs(save_dir, exist_ok=True)
    
    # load face landmark detector
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')
    # list images in the directory
    img_list = glob.glob1(img_dir, format)
    #img_list = os.listdir(img_dir)
    img_list.sort()

    # save align statistics
    stat_dict = {'quad':[], 'qsize':[], 'coord':[], 'crop':[]}

    for idx, img_name in enumerate(img_list):

        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        lm = []

        preds = fa.get_landmarks(img_np)
        for kk in range(68):
            lm.append((preds[0][kk][0], preds[0][kk][1]))
        if len(lm)==0:
            continue
        
        # Parse landmarks. Code extracted from ffhq-dataset
        # pylint: disable=unused-variable
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean([lm_eye_left[0], lm_eye_left[3]], axis=0)
        eye_right    = np.mean([lm_eye_right[0], lm_eye_right[3]], axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = np.array(lm_mouth_outer[0])
        mouth_right  = np.array(lm_mouth_outer[6])
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= np.hypot(*eye_to_eye) * 2.0#max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        stat_dict['coord'].append(quad)
        stat_dict['qsize'].append(qsize)

        qsize = stat_dict['qsize'][idx]
        quad = np.array(stat_dict['coord'][idx])
        """
        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            print('shrink!')
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink
        """
        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]
        
        stat_dict['crop'].append(crop)
        stat_dict['quad'].append((quad + 0.5).flatten())
        #img = img.crop(crop)
        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'edge')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]
        # Transform.
        img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
        img_pil = img.resize((output_size, output_size), Image.LANCZOS)
        
        # resizing
        img_pil.save(save_dir+img_name)

    np.save(save_dir+'stat_dict.npy', stat_dict)

img_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_mask(img_dir, save_dir, parsing_net, labels=[1,2,3,4,5,6,9,10,11,12,13], output_size=(1024, 1024), device=torch.device('cuda')):
    os.makedirs(save_dir, exist_ok=True)
    img_list = glob.glob1(img_dir, 'frame*')
    img_list.sort()

    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).resize((512, 512), Image.LANCZOS)
        x_1 = img_to_tensor(img).unsqueeze(0).to(device)
        out_1 = parsing_net(x_1)
        parsing = out_1[0].squeeze(0).detach().cpu().numpy().argmax(0)
        mask = np.uint8(parsing)
        for j in labels:
            mask = np.where(mask==j, 255, mask)
        mask = np.where(mask==255, 255, 0)
        mask_pil = Image.fromarray(np.uint8(mask)).resize(output_size, Image.LANCZOS)
        save_path = os.path.join(save_dir, img_name)
        mask_pil.save(save_path)