# Self-Supervised Ground-Relative Pose Estimation (ICPR 2022)
## Abstract

We propose a self-supervised method for relative 
pose estimation. Unlike existing self-supervised methods, 
we do not train a dense depth estimation network in conjunction with 
our pose network and hence avoid the complexity and ambiguity of this much 
harder problem. Instead, we use a very simple geometric model in which we 
assume the local road scene is planar. By estimating a 9D ground-relative 
pose, we are able to perform cross-projection between images via the ground 
plane using only a homography to compute a self-supervised appearance loss 
between overlapping images. We use a geometric matching architecture that 
can handle arbitrary pose pairs and use a pretrained feature extractor to 
compute a perceptual appearance loss. Our approach is competitive with 
more complex visual odometry methods that estimate dense depth maps.

## Coming Soon

- [ ] DataLoading
- [ ] Option Parsing
- [ ] Requirments
- [ ] Pretrained models
- [ ] Visual demo

This is the base code for our work on leveraging the road plane geometry
for solving self-supervised relative pose estimation using only a single
pose network. Currently you will need to supply your own dataloader, option 
parsing and logging functionality. We developed using PyTorch and PyTorch-Lightning.