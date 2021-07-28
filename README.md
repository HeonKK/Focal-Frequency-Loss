# Focal-Frequency-Loss
This repository tries to implement [Focal Frequency Loss for Image Reconstruction and Synthesis](https://arxiv.org/abs/2012.12821) by pytorch

## Reference
 [[pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)] pytorch-CycleGAN_and-pix2pix   
 [[fid](https://github.com/mseitzer/pytorch-fid)] pytorch-fid
 
## Environment
 Ubuntu 16.04   
 Pytorch 1.9.0   
 Python 3.7.10   
 Numpy 1.21.1   

## Metric
 | Model | FID |
 |:---:|:----:|
 | pix2pix(pretrained)| 128.7069 |
 | pix2pix + FFL | 125.4260 |
