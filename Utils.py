import numpy as np
import torch

def build_spatial_coordinate(L_RO, L_PE1, L_PE2):
    x = np.linspace(0, 1, L_RO)              #*********
    y = np.linspace(0, 1, L_PE1)
    z = np.linspace(0, 1, L_PE2)               #*********
    x, y, z = np.meshgrid(x, y, z,indexing='ij')  # (L, L), (L, L), (L, L)
    xyz = np.stack([x, y, z], -1).reshape(-1, 3)  # (L*L*L, 3)
    xyz = xyz.reshape(L_RO, L_PE1, L_PE2, 3)
    return xyz

def build_temporal_coordinate(numFA):
    x = np.linspace(0, 1, numFA)     #*********
    y = np.zeros((numFA))            #********* 
    xy = np.stack([x, y], -1)
    
    return xy


def BTS_signal(TR, TR_BTS,EW, EW0, T1R, T1, I0, f, k, flip_angles, eps=1e-5):
    """
    Computes the BTS signal for a given TR, EW, EW0, T1R, T1, I0, f, k and flip angle.

    Args:
    - TR  (float): repetition time                          [s]
    - TR_BTS (float): repetition time for BTS               [s]
    - EW  (torch.Tensor): 3D tensor of EW values
    - EW0 (torch.Tensor): 4D tensor of EW0 values
    - T1R (torch.Tensor): 3D tensor of T1R values           [s]
    - T1  (torch.Tensor): 3D tensor of T1 values            [s]
    - I0  (torch.Tensor): 3D tensor of I0 values
    - f   (torch.Tensor): 3D tensor of macromolecular proton fraction f values
    - k   (torch.Tensor): 3D tensor of fundamental rate constant k values
    - flip_angles (torch.Tensor): 4D tensor of flip angle   [deg]

    Returns:
    - img (torch.Tensor): 8 3D tensor of BTS signals
    """

    E1F = torch.exp(-1.0 * TR / (T1 + eps))
    E1R = torch.exp(-1.0 * TR / T1R)
    Ek  = torch.exp(-1.0 * k * TR)
    C   = f + Ek - f * Ek

    E1F_BTS = torch.exp(-1.0 * TR_BTS / (T1 + eps))
    E1R_BTS = torch.exp(-1.0 * TR_BTS / T1R)
    Ek_BTS  = torch.exp(-1.0 * k * TR_BTS)
    C_BTS   = f + Ek_BTS - f * Ek_BTS

    img_fa1 = I0 * (1.0 - f) * torch.sin(torch.deg2rad(flip_angles[0])) * (1.0 - (E1F * (1.0 - f + f * Ek - Ek * E1R * EW0[0]) + E1R * (f - f * Ek + Ek * EW0[0]))) \
                    / (1.0 - (E1F * (1.0 - f + f * Ek - Ek * E1R * EW0[0]) * torch.cos(torch.deg2rad(flip_angles[0])) + E1R * EW0[0] * C) + eps)
    
    img_fa1[torch.isnan(img_fa1)] = 0

    img_fa2 = I0 * (1.0 - f) * torch.sin(torch.deg2rad(flip_angles[1])) * (1.0 - (E1F * (1.0 - f + f * Ek - Ek * E1R * EW0[1]) + E1R * (f - f * Ek + Ek * EW0[1]))) \
                    / (1.0 - (E1F * (1.0 - f + f * Ek - Ek * E1R * EW0[1]) * torch.cos(torch.deg2rad(flip_angles[1])) + E1R * EW0[1] * C) + eps)
    
    img_fa2[torch.isnan(img_fa2)] = 0

    img_fa3 = I0 * (1.0 - f) * torch.sin(torch.deg2rad(flip_angles[2])) * (1.0 - (E1F * (1.0 - f + f * Ek - Ek * E1R * EW0[2]) + E1R * (f - f * Ek + Ek * EW0[2]))) \
                    / (1.0 - (E1F * (1.0 - f + f * Ek - Ek * E1R * EW0[2]) * torch.cos(torch.deg2rad(flip_angles[2])) + E1R * EW0[2] * C) + eps)
    
    img_fa3[torch.isnan(img_fa3)] = 0

    img_fa4 = I0 * (1.0 - f) * torch.sin(torch.deg2rad(flip_angles[3])) * (1.0 - (E1F * (1.0 - f + f * Ek - Ek * E1R * EW0[3]) + E1R * (f - f * Ek + Ek * EW0[3]))) \
                    / (1.0 - (E1F * (1.0 - f + f * Ek - Ek * E1R * EW0[3]) * torch.cos(torch.deg2rad(flip_angles[3])) + E1R * EW0[3] * C) + eps)
    
    img_fa4[torch.isnan(img_fa4)] = 0

    img_fa5 = I0 * (1.0 - f) * torch.sin(torch.deg2rad(flip_angles[0])) * (1.0 - (E1F_BTS * (1.0 - f + f * Ek_BTS - Ek_BTS * E1R_BTS * EW * EW0[0]) + E1R_BTS * (f - f * Ek_BTS + Ek_BTS * EW * EW0[0]))) \
                    / (1.0 - (E1F_BTS * (1.0 - f + f * Ek_BTS - Ek_BTS * E1R_BTS * EW * EW0[0]) * torch.cos(torch.deg2rad(flip_angles[0])) + E1R_BTS * EW0[0] * EW * C_BTS) + eps)
    
    img_fa5[torch.isnan(img_fa5)] = 0
  
    img_fa6 = I0 * (1.0 - f) * torch.sin(torch.deg2rad(flip_angles[1])) * (1.0 - (E1F_BTS * (1.0 - f + f * Ek_BTS - Ek_BTS * E1R_BTS * EW * EW0[1]) + E1R_BTS * (f - f * Ek_BTS + Ek_BTS * EW * EW0[1]))) \
                    / (1.0 - (E1F_BTS * (1.0 - f + f * Ek_BTS - Ek_BTS * E1R_BTS * EW * EW0[1]) * torch.cos(torch.deg2rad(flip_angles[1])) + E1R_BTS * EW0[1] * EW * C_BTS) + eps)
    
    img_fa6[torch.isnan(img_fa6)] = 0

    img_fa7 = I0 * (1.0 - f) * torch.sin(torch.deg2rad(flip_angles[2])) * (1.0 - (E1F_BTS * (1.0 - f + f * Ek_BTS - Ek_BTS * E1R_BTS * EW * EW0[2]) + E1R_BTS * (f - f * Ek_BTS + Ek_BTS * EW * EW0[2]))) \
                    / (1.0 - (E1F_BTS * (1.0 - f + f * Ek_BTS - Ek_BTS * E1R_BTS * EW * EW0[2]) * torch.cos(torch.deg2rad(flip_angles[2])) + E1R_BTS * EW0[2] * EW * C_BTS) + eps)
    
    img_fa7[torch.isnan(img_fa7)] = 0

    img_fa8 = I0 * (1.0 - f) * torch.sin(torch.deg2rad(flip_angles[3])) * (1.0 - (E1F_BTS * (1.0 - f + f * Ek_BTS - Ek_BTS * E1R_BTS * EW * EW0[3]) + E1R_BTS * (f - f * Ek_BTS + Ek_BTS * EW * EW0[3]))) \
                    / (1.0 - (E1F_BTS * (1.0 - f + f * Ek_BTS - Ek_BTS * E1R_BTS * EW * EW0[3]) * torch.cos(torch.deg2rad(flip_angles[3])) + E1R_BTS * EW0[3] * EW * C_BTS) + eps)
    
    img_fa8[torch.isnan(img_fa8)] = 0
    

    return img_fa1, img_fa2, img_fa3, img_fa4, img_fa5, img_fa6, img_fa7, img_fa8


