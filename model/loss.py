from __future__ import print_function, division
import torch
from torch import transpose as tp
import torchvision.models as models
from util.torch_util import rotation_composition

    
class RelativePoseLoss(torch.nn.Module):
    def __init__(self):
        super(RelativePoseLoss, self).__init__()
    
    @staticmethod
    def forward(theta , t_1to2_tar , R_1to2_tar, fix_priors=False):
        bs = theta.size()[0]

        if fix_priors:
            R_1_pred = rotation_composition(torch.zeros(3).unsqueeze(0).repeat(bs,1).cuda())
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

        R_1to2_pred = torch.matmul(R_2_pred, tp(R_1_pred, -2, -1))  # B x 3 x 3     X    B x 3 x 3
        t_1to2_pred = t_2_pred - torch.matmul(R_1to2_pred, t_1_pred)  # (B-X) x 3 x 1
        loss_t = torch.mean(torch.norm(t_1to2_pred - t_1to2_tar, dim=(1, 2)))
        loss_R = torch.mean(torch.norm(torch.matmul(R_1to2_pred, tp(R_1to2_tar, -2, -1)) - torch.eye(3).type_as(theta).unsqueeze(0).repeat(R_1to2_pred.shape[0], 1, 1), dim=(1, 2)))
        
        return (loss_R, loss_t)




class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self ,mask_norm=True , resize=True):
        super(VGGPerceptualLoss, self).__init__()

        blocks = []
        #print("layers " , models.vgg16(pretrained=True).features[:4])
        blocks.append(models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(models.vgg16(pretrained=True).features[9:16].eval())

        for bl in blocks:
            bl = bl # .to(torch.device("cuda"))
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.mask_norm = mask_norm
        
    def forward(self, inp, target, mask):  # mask where black is true
        if self.mask_norm:
            mask_norms_batchwise = torch.sum(~ mask , dim=[1,2,3] , keepdim=True).float() # need to check for max
            mask_norms_batchwise = torch.max(mask_norms_batchwise, torch.ones(1).type_as(inp))
        x = inp
        y = target

        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)

        if self.mask_norm:
            loss_batches = torch.sum((x - y)**2 , dim=[1,2,3] , keepdim=True) / mask_norms_batchwise
            # loss_batches = torch.sum(torch.abs(x - y), dim=[1, 2, 3], keepdim=True) / mask_norms_batchwise
        else:
            loss_batches = torch.sum((x - y)**2 , dim=[1,2,3] , keepdim=True)
            # loss_batches = torch.sum(torch.abs(x - y), dim=[1, 2, 3], keepdim=True)
        loss = torch.mean(loss_batches)

        return loss
    
