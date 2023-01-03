# Self-Supervised Ground-Relative Pose Estimation (ICPR 2022)
# Self-supervised Relative Pose with Homography Model-fitting in the Loop (WACV 2023)


## Abstract

We propose a self-supervised method for relative pose estimation for road scenes. By exploiting the approximate planarity of the local ground plane, we can extract a self-supervision signal via cross-projection between images using a homography derived from estimated ground-relative pose. We augment cross-projected perceptual loss by including classical image alignment in the network training loop. We use pretrained semantic segmentation and optical flow to extract ground plane correspondences between approximately aligned images and RANSAC to find the best fitting homography. By decomposing to ground-relative pose, we obtain pseudo labels that can be used for direct supervision. We show that this extremely simple geometric model is competitive for visual odometry with much more complex self-supervised methods that must learn depth estimation in conjunction with relative pose.

## Visual Results

[![Visual Results](https://img.youtube.com/vi/VrLbDH8LTFc/0.jpg)](https://www.youtube.com/watch?v=VrLbDH8LTFc)

## Coming Soon

- [ ] Homography Estimation Module
- [ ] DataLoading
- [ ] Option Parsing
- [ ] Requirments
- [ ] Pretrained models
- [ ] Visual demo

This is the base code for our work on leveraging the road plane geometry
for solving self-supervised relative pose estimation using only a single
pose network. Currently you will need to supply your own dataloader, option 
parsing and logging functionality. We developed using PyTorch and PyTorch-Lightning.
