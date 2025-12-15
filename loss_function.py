import torch
import torch.nn as nn

class MYTVLoss(nn.Module):
    def __init__(self):
        super(MYTVLoss, self).__init__()

    def forward(self, x):
        N1, N2, N3 = x.shape[0], x.shape[1], x.shape[2]
        ndim = x.dim()
        if ndim == 3:
            tv_loss = (torch.sum(torch.abs(x[:, 1:, :] - x[:, :N2-1, :])) )/ ((N2-1))
        else:
            tv_loss = (torch.sum(torch.abs(x[:, 1:, :, :] - x[:, :N2-1, :, :])))/ ((N2-1))
        
        return tv_loss

class MYTVLoss_3D(nn.Module):
    def __init__(self):
        super(MYTVLoss_3D, self).__init__()

    def forward(self, x):
        N1, N2, N3 = x.shape[0], x.shape[1], x.shape[2]
        ndim = x.dim()
        if ndim == 3:
            tv_loss = (torch.sum(torch.abs(x[1:, :, :] - x[:N1-1, :, :]))+torch.sum(torch.abs(x[:, 1:, :] - x[:, :N2-1, :]))+torch.sum(torch.abs(x[:,:,1:] - x[:,:,:N3-1])) )/ ((N1-1)*(N2-1)*(N3-1))
        else:
            tv_loss = (torch.sum(torch.abs(x[1:, :, :, :] - x[:N1-1, :, :, :]))+torch.sum(torch.abs(x[:, 1:, :, :] - x[:, :N2-1, :, :]))+torch.sum(torch.abs(x[:,:,1:, :] - x[:,:,:N3-1, :])) )/ ((N1-1)*(N2-1)*(N3-1))
        
        return tv_loss

MAE_loss_function = torch.nn.L1Loss()

def compute_cost_ksp(output, target, sensitivity):

    cost = 0.
    for img in output:
        output_ksp   = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift((img.unsqueeze(3))*sensitivity,dim=(0,1,2)), dim=(0,1,2)), dim=(0,1,2))  
        cost         +=  MAE_loss_function(torch.view_as_real(output_ksp.masked_select(target!=0)).float(), 
                                            torch.view_as_real(target.masked_select(target!=0)).float())
    return cost
    

