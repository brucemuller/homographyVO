import shutil
import torch
from os import makedirs
from os.path import exists, join, basename, dirname
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw
from torchvision.transforms import functional as tfn
from torchvision import transforms
from torchvision.utils import make_grid
import decimal
import pickle
from csv import reader, writer
import argparse
ctx = decimal.Context()
ctx.prec = 20

def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

def save_checkpoint(state, is_best, file, epoch_save=False, epoch=-1):
    model_dir = dirname(file)
    model_name = basename(file)
    # make dir if needed (should be non-empty)
    if model_dir!='' and not exists(model_dir):
        makedirs(model_dir)
    torch.save(state, file)
    if is_best:
        shutil.copyfile(file, join(model_dir,'best_' + model_name))
    if epoch_save:
        altered_filename = join(model_dir, 'epoch' + str(epoch) + '_' + model_name)
        torch.save(state, altered_filename)
        
def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def expand_dim(tensor,dim,desired_dim_len):
    sz = list(tensor.size())
    sz[dim]=desired_dim_len
    return tensor.expand(tuple(sz))


class UnNormalize(object):
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
#        if len(tensor.shape)==3:
#            tensor = tensor.unsqueeze(0)
        tensor_clone = tensor.clone().detach()
        bs = tensor.shape[0]
        for x in range(bs):
            tensor_be = tensor_clone[x,:,:,:].clone()
            for t, m, s in zip(tensor_be, self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
            tensor_clone[x,:,:,:] = tensor_be
        return tensor_clone
    




def uppercut(img):
    h = int(img.size()[2] / 2)
    img = img.clone()
    img[:,:,0:h,:] = 0.0
    return img

def uppercut_m(mask):
    h = int(mask.size()[2] / 2)
    mask = mask.clone()
    mask[:,:,0:h,:] = True
    return mask

def warp_image(imgs, H, border_padding=False, mode='bilinear'):
    h = imgs.shape[-2]
    w = imgs.shape[-1]

    H = H.reshape(-1, 3, 3)

    # Generate regular grid
    xs = torch.linspace(-1, 1, w).type_as(H)  # returns 1D tensor 1,2,3,4,...,800
    ys = torch.linspace(-1, 1, h).type_as(H)  # h
    grid_x, grid_y = torch.meshgrid([xs, ys])
    # Reshape to 3 x n array where 3rd row is composed of ones
    X = grid_x.flatten()
    Y = grid_y.flatten()
    xy = torch.stack((X, Y, torch.ones(h * w).type_as(H))).unsqueeze(0).repeat(H.size()[0], 1, 1)
    xyt = torch.matmul(H, xy)  # result: 3 x 512000 where each column is a point in transformed coords

    xyt[:, 2, :] = xyt[:, 2, :] + 0.0000001
    xyt = xyt / (xyt[:, 2, :].unsqueeze(1))  # CHECK THIS

    Xt = torch.transpose(xyt[:, 0, :].view(H.size()[0], w, h), 1, 2)  # takes transformed X coords, reshape to 640 x 800 tensor. VIEW WARNING
    Yt = torch.transpose(xyt[:, 1, :].view(H.size()[0], w, h), 1, 2)  # similarly for Y
    grid = torch.cat((Xt.unsqueeze(-1), Yt.unsqueeze(-1)), dim=-1)  # .repeat(H.size[0],1,1,1)  # B_masked, 640, 800, 2

    if border_padding:
        warped_img_right = F.grid_sample(imgs, grid, mode=mode, padding_mode='border')  # in: input=(N x C x H_in x W_in)  grid=(N x H_out x W_out x 2) ,  out: N x C x H_out x W_out
    else:
        warped_img_right = F.grid_sample(imgs, grid, mode=mode, padding_mode='zeros')  # in: input=(N x C x H_in x W_in)  grid=(N x H_out x W_out x 2) ,  out: N x C x H_out x W_out

    return warped_img_right

def log_stats(stats_dict, tb_writer, epoch, batch_idx, dataloader_size):
    for value_name in stats_dict:
        tb_writer.add_scalar(value_name,stats_dict[value_name].data.item(), (epoch - 1) * dataloader_size + batch_idx)

def rotation_composition(rpy_vec):
    gamma = rpy_vec[:, 0];
    cos_gamma = torch.cos(gamma).unsqueeze(1)
    sin_gamma = torch.sin(gamma).unsqueeze(1)
    beta = rpy_vec[:, 1];
    cos_beta = torch.cos(beta).unsqueeze(1)
    sin_beta = torch.sin(beta).unsqueeze(1)
    alpha = rpy_vec[:, 2];
    cos_alpha = torch.cos(alpha).unsqueeze(1)
    sin_alpha = torch.sin(alpha).unsqueeze(1)

    zeros = torch.zeros(rpy_vec.shape[0]).type_as(rpy_vec).unsqueeze(1)
    ones  = torch.ones(rpy_vec.shape[0]).type_as(rpy_vec).unsqueeze(1)

    top = torch.cat((cos_gamma, -1 * sin_gamma, zeros), dim=1)
    middle = torch.cat((sin_gamma, cos_gamma, zeros), dim=1)
    bottom = torch.cat((zeros, zeros, ones), dim=1)
    roll = torch.cat((top.unsqueeze(1), middle.unsqueeze(1), bottom.unsqueeze(1)), dim=1)

    top = torch.cat((ones, zeros, zeros), dim=1)
    middle = torch.cat((zeros, cos_beta, -1 * sin_beta), dim=1)
    bottom = torch.cat((zeros, sin_beta, cos_beta), dim=1)
    pitch = torch.cat((top.unsqueeze(1), middle.unsqueeze(1), bottom.unsqueeze(1)), dim=1)

    top = torch.cat((cos_alpha, zeros, sin_alpha), dim=1)
    middle = torch.cat((zeros, ones, zeros), dim=1)
    bottom = torch.cat((-1 * sin_alpha, zeros, cos_alpha), dim=1)
    yaw = torch.cat((top.unsqueeze(1), middle.unsqueeze(1), bottom.unsqueeze(1)), dim=1)

    # Rotate to canonical camera pose
    R_LtoCam = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).type_as(rpy_vec).unsqueeze(0).repeat(rpy_vec.shape[0], 1, 1)

    return torch.matmul(roll, torch.matmul(pitch, torch.matmul(yaw, R_LtoCam)))