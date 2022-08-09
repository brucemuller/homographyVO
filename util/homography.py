import torch.nn as nn
import torch

from util.torch_util import rotation_composition

class GroundPlaneRepToHomography(nn.Module):
    def __init__(self, fix_priors):
        super(GroundPlaneRepToHomography, self).__init__()
        self.fix_priors = fix_priors

    # Compose rotation matrix from roll, pitch and yaw.


    def forward(self, theta, Ki, Kj, T_1241_1):

        bs = theta.size()[0]

        if self.fix_priors:
            R_1_pred = rotation_composition(torch.zeros(3).unsqueeze(0).repeat(bs, 1).cuda())
            R_2_pred = rotation_composition(torch.cat((torch.zeros(1).unsqueeze(0).repeat(bs, 2).cuda(), theta[:, -1].unsqueeze(-1)), dim=-1))
            c_Cam1 = torch.cat((torch.zeros(2).type_as(theta).unsqueeze(0).repeat(bs, 1), torch.cuda.FloatTensor([1.65]).unsqueeze(0).repeat(bs, 1)), dim=1)
            c_Cam2 = torch.cat((theta[:, 0:2], torch.cuda.FloatTensor([1.65]).unsqueeze(0).repeat(bs, 1)), dim=-1)  # B x 3
            t_1_pred = -1 * torch.matmul(R_1_pred, c_Cam1.unsqueeze(-1))  # - Rc   ....      B x 3 x 3    X    B x 3 x 1   =    B x 3 x 1
            t_2_pred = -1 * torch.matmul(R_2_pred, c_Cam2.unsqueeze(-1))  # B x 3 x 1
        else:
            R_1_pred = rotation_composition(torch.cat((theta[:, 1:3], torch.zeros(1).type_as(theta).unsqueeze(0).repeat(theta.shape[0], 1)), dim=-1))
            R_2_pred = rotation_composition(theta[:, 6:9])
            c_Cam1 = torch.cat((torch.zeros(2).type_as(theta).unsqueeze(0).repeat(theta.shape[0], 1), theta[:, 0].unsqueeze(-1)), dim=1)
            t_1_pred = -1 * torch.matmul(R_1_pred, c_Cam1.unsqueeze(-1))  # - Rc   ....      B x 3 x 3    X    B x 3 x 1   =    B x 3 x 1
            t_2_pred = -1 * torch.matmul(R_2_pred, theta[:, 3:6].unsqueeze(-1))  # B x 3 x 1


        Rt_1 = torch.cat((R_1_pred, t_1_pred), dim=-1)
        Rt_2 = torch.cat((R_2_pred, t_2_pred), dim=-1)

        Rt_1 = torch.cat((Rt_1[:, :, :2], Rt_1[:, :, 3].unsqueeze(-1)), -1)  # remove 3rd column
        T1 = torch.matmul(Ki, Rt_1)  # K[R t]

        Rt_2 = torch.cat((Rt_2[:, :, :2], Rt_2[:, :, 3].unsqueeze(-1)), -1)  # remove 3rd column,
        T2 = torch.matmul(Kj, Rt_2)

        H = torch.matmul(T2, torch.inverse(T1))

        # convert homography to -1,1 space for warping compatibility with PyTorch grid sampler
        H_postmultiplied = torch.matmul(H, torch.inverse(T_1241_1))
        H_premultiplied = torch.matmul(T_1241_1, H_postmultiplied)

        return H_premultiplied