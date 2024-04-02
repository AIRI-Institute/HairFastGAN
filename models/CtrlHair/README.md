# [GAN with Multivariate Disentangling for Controllable Hair Editing (ECCV 2022)](https://github.com/XuyangGuo/xuyangguo.github.io/raw/master/database/CtrlHair/CtrlHair.pdf)

[Xuyang Guo](https://xuyangguo.github.io/), [Meina Kan](http://vipl.ict.ac.cn/homepage/mnkan/Publication/), Tianle Chen, [Shiguang Shan](https://scholar.google.com/citations?user=Vkzd7MIAAAAJ)

![demo1](https://github.com/XuyangGuo/xuyangguo.github.io/blob/master/database/CtrlHair/resources/demo1.gif?raw=true)

![demo2](https://github.com/XuyangGuo/xuyangguo.github.io/blob/master/database/CtrlHair/resources/demo2.gif?raw=true)

![demo3](https://github.com/XuyangGuo/xuyangguo.github.io/blob/master/database/CtrlHair/resources/demo3.gif?raw=true)

## Abstract

> Hair editing is an essential but challenging task in portrait editing considering the complex geometry and material of hair. Existing methods have achieved promising results by editing through a reference photo, user-painted mask, or guiding strokes. However, when a user provides no reference photo or hardly paints a desirable mask, these works fail to edit. Going a further step, we propose an efficiently controllable method that can provide a set of sliding bars to do continuous and fine hair editing. Meanwhile, it also naturally supports discrete editing through a reference photo and user-painted mask. Specifically, we propose a generative adversarial network with a multivariate Gaussian disentangling module. Firstly, an encoder disentangles the hair's major attributes, including color, texture, and shape, to separate latent representations. The latent representation of each attribute is modeled as a standard multivariate Gaussian distribution, to make each dimension of an attribute be changed continuously and finely. Benefiting from the Gaussian distribution, any manual editing including sliding a bar, providing a reference photo, and painting a mask can be easily made, which is flexible and friendly for users to interact with. Finally, with changed latent representations, the decoder outputs a portrait with the edited hair. Experiments show that our method can edit each attribute's dimension continuously and separately. Besides, when editing through reference images and painted masks like existing methods, our method achieves comparable results in terms of FID and visualization. Codes can be found at [https://github.com/XuyangGuo/CtrlHair](https://github.com/XuyangGuo/CtrlHair).

![archi](https://raw.githubusercontent.com/XuyangGuo/xuyangguo.github.io/master/database/CtrlHair/resources/architecture.png)


## Installation

Clone this repo.

```bash
git clone https://github.com/XuyangGuo/CtrlHair.git
cd CtrlHair
```

The code requires python 3.6.

We recommend using Anaconda to regulate packages.

Dependencies:
- PyTorch 1.8.2
- torchvision, tensorboardX, dlib
- pillow, pandas, scikit-learn, opencv-python
- PyQt5, tqdm, addict, dill

Please download all [external trained models](https://drive.google.com/drive/folders/1X0y82o7-JB6nGYIWdbbJa4MTFbtV9bei?usp=sharing) ([using Baidu Netdisk](https://pan.baidu.com/s/1vnldtoT9G-5gfvUjjUSVOA?pwd=1234) with password `1234` alternatively), move it to `./` with the correct path `CtrlHair/external_model_params`. (The directory contains the model parameters of face analysis tools required by our project, including SEAN for face encoding and generator with masks, BiSeNet for face parsing, and 68/81 facial landmark detector.)

## Editing by Pretrained Model

Firstly, refer to the "Installation" section above.

Please download [the pre-trained model of CtrlHair](https://drive.google.com/drive/folders/1opQhmc7ckS3J8qdLii_EMqmxYCcBLznO?usp=sharing) ([using Baidu Netdisk](https://pan.baidu.com/s/1O_Hu5dnk4GwmUrIBDWpwFA?pwd=1234) with password `1234` alternatively), move it to `./` with the correct path `CtrlHair/model_trained`.

#### Editing with UI directly (recommended)

```bash
python ui/frontend_demo.py
```

Here are some parameters of it:

- `-g D` Use the gpu `D`. (default is `0`)
- `-n True|False` Whether the input image need crop. (default is `True`)
- `--no_blending` Do not use poisson blending as post processing. If not blend, the result image will look slightly different from the input image in some details in non-hair region, but the image quality will be better.

The edited results can be found in `temp_folder/demo_output/out_img.png`, and the edited shape is in `temp_folder/demo_output/input_parsing.png`. The `temp_folder` is created automaticly during running, which could be removed after closing the procedure.

#### Editing with Batch

If you want to edit with a mass batch, or want to achieve editing functions such as interpolation, multi style sampling, and continuous gradient, etc, please use the interfaces of `ui/backend.py/Backend` and code your own python script.`Backend` class is the convenient encapsulation of basic functions of CtrlHair, and there are detailed comments for each function. The `main` scetion in the final of `backend.py` shows an simle example of usage of `Backend`.

## Training New Models

#### Data Preparation
In addition to the images in the dataset, training also involves many image annotations, including face parsing, facial landmarks, sean codes, color annotations, face rotation angle, a small amount of annotations (Curliness of Hair), etc.

Please download [the dataset information](https://drive.google.com/drive/folders/10p87Mobgueg9rdHyLPkX8xqEIvga6p2C?usp=sharing)  ([using Baidu Netdisk](https://pan.baidu.com/s/1D3D2JqxIR6miCeMNssucKw?pwd=1234) with password `1234` alternatively) that we have partially processed, move it to `./` with the correct path `CtrlHair/dataset_info_ctrlhair`.

Then execute the following scripts sequentially for preprocessing.

Get facial segmentation mask
```bash
python dataset_scripts/script_get_mask.py
```

Get 68 facial landmarks and 81 facial landmarks
```bash
python dataset_scripts/script_landmark_detection.py
```

Get SEAN feature codes of the dataset
```bash
python dataset_scripts/script_get_sean_code.py
```

Get color label of the dataset
```bash
python dataset_scripts/script_get_rgb_hsv_label.py
python dataset_scripts/script_color_var_label.py
```

After complete processing, for training, the correct directory structure in `CtrlHair/dataset_info_ctrlhair` is as follows:

- `CelebaMask_HQ` / `ffhq`  (if you want to add your own dataset, please regulate them as these two cases)
  - `images_256`   -> cropped images with the resolution 256.
  - `label`       -> mask label (0, 1, 2, ..., 20) for each pixel
  - `angle.csv`    restore face rotation angle of each image
  - `attr_gender.csv`  restore gender of each image
- `color_var_stat_dict.pkl`, `rgb_stat_dict.pkl`, `hsv_stat_dict_ordered.pkl` store the label of variance, rgb of hair color, and the hsv distribution
- `sean_code_dict.pkl`  store the sean feature code of images in dataset
- `landmark68.pkl`, `landmark81.pkl`  store the facial landmarks of the dataset
- `manual_label`
  - `curliness`
    - `-1.txt`, `1.txt`, `test_1.txt`, `test_-1.txt` labeled data list

#### Training Networks
In order to better control the parameters in the model, we train the entire model separately and divide it into four parts, including curliness classifier, color encoder, color & texture branch, shape branch.

**1. Train the curliness classifier**
```bash
python color_texture_branch/predictor/predictor_train.py -c p002 -g 0
```

Here are some parameters of it:

- `-g D` Use the gpu `D`. (default is `0`)
- `-c pxxx` Using the model hyper-parameters config named `pxxx`. Please see the config detail in `color_texture_branch/predictor/predictor_config.py`

The trained model and its tensorboard summary are saved in `model_trained/curliness_classifier`.

**2. Train the color encoder**
```bash
python color_texture_branch/predictor/predictor_train.py -c p004 -g 0
```

The parameters are similar like the curliness classifier.

The trained model and its tensorboard summary are saved in `model_trained/color_encoder`.

**3. Train the color & texture branch**
```bash
python color_texture_branch/scripts.py -c 045 -g 0
```

This part depends on the curliness classifier and the color encoder as seen in `color_texture_branch/config.py`:

```python
...
'predictor': {'curliness': 'p002', 'rgb': 'p004'},
...
```

Here are some parameters of it:

- `-g D` Use the gpu `D`. (default is `0`)
- `-c xxx` Using the model hyper-parameters config named `xxx`. Please see the config detail in `color_texture_branch/config.py`

The trained model, its tensorboard, editing results in training are saved in `model_trained/color_texture`.


Since the training of texture is unsupervised, we need to find some semantic orthogonal directions after training for sliding bars. Please run:
```bash
python color_texture_branch/script_find_direction.py -c xxx
```
The parameter `-c xxx` is same as above shape config.
This process will generate a folder named `direction_find` in the directory `model_trained/color_texture/yourConfigName`,
where `direction_find/texture_dir_n` stores many random `n`-th directions to be selected, 
and the corresponding visual changes can be seen in `direction_find/texture_n`. 
When the choice is decided, move the corresponding `texture_dir_n/xxx.pkl` file to `../texture_dir_used` and rename it as you wish (the pretrained No.045 texture model shows an example).
Afterthat, run `python color_texture_branch/script_find_direction.py -c xxx` again, and repeat the process until the amount of semantic directions is enough.


**4. Train the shape branch**

Shape editing employs a transfer-like training process. Before transferring, we use 68 feature points to pre-align the face of the target hairstyle, so as to achieve a certain degree of face adaptation. In order to speed up this process during training, it is necessary to generate some buffer pools to store the pre-aligned masks of the training set and test set respectively.

Generate buffer pool for testing
```bash
python shape_branch/script_adaptor_test_pool.py
```
The testing buffer pool will be saved in `dataset_info_ctrlhair/shape_testing_wrap_pool`.

Generate buffer pool for training
```bash
python shape_branch/script_adaptor_train_pool.py
```
The training buffer pool will be saved in `dataset_info_ctrlhair/shape_training_wrap_pool`.

Note that the `script_adaptor_train_pool.py` process will execute for a very very long time until the setting of maximum number of files for buffering is reached.  
This process can be performed concurrently with the subsequent shape training process. 
The training data for shape training are all dynamically picked from this buffer pool.

Training the shape branch model
```bash
python shape_branch/scripts.py -c 054 -g 0
```

Here are some parameters of it:

- `-g D` Use the gpu `D`. (default is `0`)
- `-c xxx` Using the model hyper-parameters config named `xxx`. Please see the config detail in `shape_branch/config.py`

The trained model, its tensorboard, editing results in training are saved in `model_trained/shape`.

Since the training is unsupervised, we need to find some semantic orthogonal directions after training for sliding bars. Please run:
```bash
python shape_branch/script_find_direction.py -c xxx
```
The parameter `-c xxx` is same as above shape config.
The entire usage method is similar to texture, but the folder is changed to the `model_trained/shape` directory (the pretrained No.054 shape model shows an example).

After all the above training, use `python ui/frontend_demo.py` to edit. 
You can also use interfaces in `ui/backend.py/Backend` to program your editing scripts.

## Training New Models with Your Own Images Dataset

Our method only needs unlabeled face images to augment the dataset, which is convenient and is a strength of CtrlHair.

#### Data Preparation

For your own images dataset, firstly, crop and resize them. Please collect them into a single directory, and modify `root_dir` and `dataset_name` for your dataset in `dataset_scripts/script_crop.py`. Then execute
```bash
python dataset_scripts/script_crop.py
```
After cropping, the dataset should be cropped at `dataset_info_ctrlhair/your_dataset_name/images_256`. Your dataset should have similar structure like `dataset_info_ctrlhair/ffhq`.

Modify `DATASET_NAME` in `global_value_utils.py ` for your dataset.

Do the same steps as the section "Data Preparation" of "Training New Models" in this README.

Predict face rotation angle and gender for your dataset. 
This will be used to filter the dataset. 
You can use tools like [3DDFA](https://github.com/cleardusk/3DDFA) and [deepface](https://github.com/serengil/deepface), then output them to `angle.csv` and `attr_gender.csv` in `dataset_info_ctrlhair/yourdataset` (pandas is recommended for generating csv). `dataset_info_ctrlhair/ffhq` shows a preprocessing example. 
Sorry for that we don't provide these code. Alternatively, if you don't want to depend and use these filter, please modify `angle_filter` and `gender_filter` to `False` in `common_dataset.py`. 

#### Training Networks

Add and adjust your config in `color_texture_branch/predictor/predictor_config.py`, `color_texture_branch/config.py`, `shape/config.py`.

Do the same steps as the section "Training Networks" of "Training New Models" in this README, but with your config.

Finally, change the `DEFAULT_CONFIG_COLOR_TEXTURE_BRANCH` and `DEFAULT_CONFIG_SHAPE_BRANCH` as yours in `global_value_utils.py`.
Use `python ui/frontend_demo.py` to edit. Or you can also use interfaces in `ui/backend.py/Backend` to program your editing scripts.


## Code Structure
- `color_texture_branch`: color and texture editing branch
  - `predictor`: color encoder and curliness classifier
- `shape_branch`: shape editing branch
- `ui`: encapsulated backend interfaces and frontend UI
- `dataset_scripts`: scripts for preprocessing dataset
- `external_code`: codes of external tools
- `sean_codes`: modified from SEAN project, which is used for image feature extraction and generation
- `my_pylib`, `my_torchlib`, `utils`: auxiliary code library
- `wrap_codes`: used for shape align before shape transfer
- `dataset_info_ctrlhair`: the root directory of dataset
- `model_trained`: trained model parameters, tensorboard and visual results during training
- `external_model_params`: pretrained model parameters used for external codes
- `imgs`: some example images are provided for testing

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{guo2022gan,
  title={GAN with Multivariate Disentangling for Controllable Hair Editing},
  author={Guo, Xuyang and Kan, Meina and Chen, Tianle and Shan, Shiguang},
  booktitle={European Conference on Computer Vision},
  year={2022},
  pages={655--670},
  organization={Springer}
}
```

This work is also inspired by our previous work [X. Guo, et al., STD-GAN (CVIU2021)](https://github.com/XuyangGuo/STD-GAN) for instance-level facial attributes editing.

## References & Acknowledgments
- [ZPdesu / SEAN](https://github.com/ZPdesu/SEAN)
- [zhhoper / RI_render_DPR](https://github.com/zhhoper/RI_render_DPR)
- [zllrunning / face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
