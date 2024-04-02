#!/bin/sh
pip install gdown
mkdir pretrained_models
cd pretrained_models

# download pretrained encoder
gdown --fuzzy https://drive.google.com/file/d/1RnnBL77j_Can0dY1KOiXHvG224MxjvzC/view?usp=sharing

# download ArcFace pretrained model
gdown --fuzzy https://drive.google.com/file/d/1coFTz-Kkgvoc_gRT8JFzqCgeC3lAFWQp/view?usp=sharing

# download face parsing model from https://github.com/zllrunning/face-parsing.PyTorch
gdown --fuzzy https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812

# download pSp pretrained model from https://github.com/eladrich/pixel2style2pixel.git
cd ../pixel2style2pixel
mkdir pretrained_models
cd pretrained_models
gdown --fuzzy https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view?usp=sharing
cd ..
cd ..

