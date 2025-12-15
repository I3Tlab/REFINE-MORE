import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import Utils
import scipy.io as sio
import os
from unet import UNet
import tinycudann as tcnn
from tqdm import tqdm
import loss_function as lf


sigmoid = nn.Sigmoid()
    
class Unrolled_PG(torch.nn.Module):
    def __init__(self,device):
        super(Unrolled_PG, self).__init__()

        self.a_step   = nn.Parameter(torch.Tensor([0.001])).to(device)
        self.b_step   = nn.Parameter(torch.Tensor([0.001])).to(device)
        self.c_step   = nn.Parameter(torch.Tensor([0.001])).to(device)

################################################################################################  
        self.Unet     = UNet(in_channels=3, out_channels=3, bilinear=True).to(device)
        
################################################################################################
    def least_squares(self, mask, phase, k_space, sensitivity, TR, TR_bts, EW, EW0, T1R, T1, I0, f, k, flip_angles):
        img = torch.stack(Utils.BTS_signal(TR, TR_bts, EW, EW0, T1R, T1, I0, f, k, flip_angles), dim=-1) # [FE, PE1, PE2, FA]
        img = torch.complex(img*torch.cos(phase),img*torch.sin(phase))
        coil_img = sensitivity * img.unsqueeze(3) # [FE,PE1,PE2,coil,FA]

        Fk = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(coil_img,dim=(0,1,2)), dim=(0,1,2)), dim=(0,1,2))*mask   
        ls = 0.5*torch.mean(torch.abs(Fk - k_space)**2)
        return ls
       
    def dlsdVar(self, mask, phase, I0, T1, f, k, k_space, sensitivity, flip_angles, TR, TR_bts, EW, EW0, T1R):
        I0 = I0.clone().detach().requires_grad_(True)
        T1 = T1.clone().detach().requires_grad_(True) 
        f  = f.clone().detach().requires_grad_(True) 
        ls = self.least_squares(mask, phase, k_space, sensitivity, TR, TR_bts, EW, EW0, T1R, T1, I0, f, k, flip_angles)
        dls_dVar = torch.autograd.grad(ls, [I0, T1, f])
        dls_dI0 = dls_dVar[0]
        dls_dT1 = dls_dVar[1]
        dls_df  = dls_dVar[2]
        return dls_dI0, dls_dT1, dls_df

    def forward(self, mask, phase, I0, T1, f, k, k_space, sensitivity, flip_angles, TR, TR_bts,EW, EW0, T1R):
        dls_dI0, dls_dT1, dls_df = self.dlsdVar(mask, phase, I0, T1, f, k,k_space, sensitivity, flip_angles, TR, TR_bts, EW, EW0, T1R)
       
        I0 = I0 - self.a_step * dls_dI0   # [FE,PE1,PE2]
        T1 = T1 - self.b_step * dls_dT1   # [FE,PE1,PE2]
        f = f - self.c_step * dls_df      # [FE,PE1,PE2]

        inp_params = torch.stack([I0.permute(2,0,1), T1.permute(2,0,1), f.permute(2,0,1)], dim=1)
        out_params = self.Unet(inp_params)
        I0_out = F.relu(out_params[:,0,:,:].permute(1,2,0))
        T1_out = F.relu(out_params[:,1,:,:].permute(1,2,0))
        f_out  = torch.sigmoid(out_params[:,2,:,:].permute(1,2,0))

        img = torch.stack(Utils.BTS_signal(TR, TR_bts, EW, EW0, T1R, T1_out, I0_out, f_out, k, flip_angles), dim=-1) # [FE, PE1, PE2, FA]
        img = torch.complex(img*torch.cos(phase),img*torch.sin(phase))

        return I0_out, T1_out, f_out, k,img
    
class INR_initialization_module(torch.nn.Module):
    def __init__(self,nFE,nPE1,nPE2,nFA,device,seedd=3598):
        super(INR_initialization_module, self).__init__()

        encoding_config={
            "otype": "Grid",
            "type": "Hash",
            "n_levels":16,
            "n_features_per_level": 2,      #default 2     
            "log2_hashmap_size": 19,        #default 19
            "base_resolution": 16,          #default 16
            "per_level_scale": 2.0,         #default 2
            "interpolation":"Linear"
            }
        network_config={
            "otype": "FullyFusedMLP",           # Compwonent type.
            "activation": "ReLU",               # Activation of hidden layers.
            "output_activation": "None",        # Activation of the output layer.
            "n_neurons":64,                     # Neurons in each hidden layer.
            "n_hidden_layers":3,                # Number of hidden layers.                        
            }
        self.nFE = nFE
        self.nPE1 = nPE1
        self.nPE2 = nPE2
        self.nFA = nFA

        ### build coordinate
        temporal_coor = Utils.build_temporal_coordinate(nFA)
        spatial_coor  = Utils.build_spatial_coordinate(L_RO=nFE, L_PE1=nPE1, L_PE2=nPE2) 
        self.temporal_coor = torch.tensor(temporal_coor).to(device)
        self.spatial_coor  = torch.tensor(spatial_coor).to(device)

        # INR representation for the weighted images
        self.Hash_enc1 = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_config).to(device)  # corresponding to the hash encoding 1 in the paper
        self.Hash_enc2 = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_config).to(device)  # corresponding to the hash encoding 2 in the paper
        self.MLPnet1   = tcnn.Network(n_input_dims=(self.Hash_enc1.n_output_dims+self.Hash_enc2.n_output_dims), n_output_dims=2, network_config=network_config,seed=seedd).to(device) # corresponding to the MLP1 in the paper

        # INR representation for the quantitative parameters
        self.Hash_enc3 = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_config).to(device)  # corresponding to the hash encoding 3 in the paper
        self.MLPnet2   = tcnn.Network(n_input_dims=(self.Hash_enc3.n_output_dims+8), n_output_dims=4, network_config=network_config,seed=seedd).to(device) # corresponding to the MLP2 in the paper


    def forward(self, mask, kspace, sensitivity,tissue_mask):
        spatial_enc    = self.Hash_enc2(self.spatial_coor.view(-1,3)).view(self.nFE,self.nPE1,self.nPE2,-1).unsqueeze(3).repeat(1,1,1,8,1)
        temporal_enc   = self.Hash_enc1(self.temporal_coor).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(self.nFE,self.nPE1,self.nPE2,1,1)

        all_enc        = torch.cat([spatial_enc, temporal_enc], dim=-1)  # Concatenate spatial and temporal encodings
        weighted_out   = self.MLPnet1(all_enc.view(-1,all_enc.shape[-1])).view(self.nFE,self.nPE1,self.nPE2,self.nFA,2).float()

        weighted_cplx  = torch.complex(weighted_out[:,:,:,:,0], weighted_out[:,:,:,:,1])  # complex weighted images

        pred_ksp       = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(weighted_cplx.unsqueeze(3)*sensitivity,dim=(0,1,2)), dim=(0,1,2)), dim=(0, 1,2))

        pred_ksp_dc   = pred_ksp*(1-mask) + kspace*mask     #[FE, PE1, PE2, Coil, FA]
        pred_weighted_sos = torch.sum(torch.abs(torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(pred_ksp_dc,dim=(0,1,2)), dim=(0,1,2)), dim=(0, 1,2)))**2,dim=3)**0.5   
# 
        initial_params_encoding = self.Hash_enc3(self.spatial_coor.view(-1,3)).view(self.nFE,self.nPE1,self.nPE2,-1).float() # [FE, PE1, PE2, 4]
        initial_params  = self.MLPnet2(torch.cat([initial_params_encoding, pred_weighted_sos.abs()], dim=-1).view(-1,40)).view(self.nFE,self.nPE1,self.nPE2,-1).float() # [FE, PE1, PE2, 4]
        initial_params  = initial_params * (tissue_mask.unsqueeze(-1))
        initial_I0      = initial_params[:,:,:,0].abs() #[FE, PE1, PE2]
        initial_T1      = initial_params[:,:,:,1].abs()
        initial_f       = torch.sigmoid(initial_params[:,:,:,2]) 
        initial_k       = torch.tanh(initial_params[:,:,:,3]) + torch.tensor(23.0).to(initial_params) 

        return weighted_cplx, initial_I0, initial_T1, initial_f, initial_k
   
        

class Physics_Reinforcement_module(torch.nn.Module):
    def __init__(self, PhaseNo,device):
        super(Physics_Reinforcement_module, self).__init__()
        self.PhaseNo = PhaseNo
        self.fcs = Unrolled_PG(device)

    def forward(self, mask, phase, k_space, sensitivity, flip_angles, TR, TR_bts, EW, EW0, T1R, T1_initial, I0_initial, f_initial, k_initial):
        img_list    = []
        I0_list     = []
        T1_list     = []
        f_list      = []
        k_list      = []

        I0          = I0_initial
        T1          = T1_initial
        f           = f_initial
        k           = k_initial

        for i in range(self.PhaseNo):
            I0, T1, f, k, img = self.fcs(mask, phase, I0, T1, f, k, k_space, sensitivity, flip_angles, TR, TR_bts, EW, EW0, T1R) # [PE2, PE1, FE]/[FA, PE2, PE1, FE]
            img_list.append(img)
            I0_list.append(I0)
            T1_list.append(T1)
            f_list.append(f)
            k_list.append(k)

        return I0_list, T1_list, f_list, k_list, img_list



def REFINE_MORE(kspace_real, kspace_imag, sensitivity_real,sensitivity_imag,mask,tissue_mask,TR,TR_BTS,EW,EW0,T1R,flip_angles,log_dir,model_dir,result_dir,device,learning_rate=1e-3,phase_num=4):
    
    kspace = torch.complex(kspace_real, kspace_imag)
    sensitivity = torch.complex(sensitivity_real, sensitivity_imag)
    
    nFE, nPE1, nPE2, nFA = kspace.shape[0], kspace.shape[1], kspace.shape[2], kspace.shape[4]

    writer = SummaryWriter(log_dir)
    
    INR_init_model = INR_initialization_module(nFE,nPE1,nPE2,nFA,device)
    Physics_refine_model = Physics_Reinforcement_module(phase_num,device)
    
    optimizer   = torch.optim.Adam([                        
                                  {'params': INR_init_model.parameters(), 'lr': learning_rate},
                                  {'params': Physics_refine_model.parameters(), 'lr': learning_rate},
                                
                         ])
    
    ### loss function
    MAE_loss_function = torch.nn.L1Loss()
    TV_loss_function  = lf.MYTVLoss()
    
    def compute_cost_ksp(output, target, sensitivity):
        cost = 0.
        for img in output:
            output_ksp   = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift((img.unsqueeze(3))*sensitivity,dim=(0,1,2)), dim=(0,1,2)), dim=(0,1,2))  
            cost         +=  MAE_loss_function(torch.view_as_real(output_ksp.masked_select(target!=0)).float(), 
                                                torch.view_as_real(target.masked_select(target!=0)).float())
        return cost
    
    '''starting reconstruction'''
    print('Reconstruction start...')
    iter_loop = tqdm(range(3000))
    loss_train = 0.0
    for e in iter_loop:
        INR_init_model.train()
        Physics_refine_model.train()

        weighted_cplx, initial_I0, initial_T1, initial_f, initial_k = INR_init_model(mask, kspace, sensitivity,tissue_mask)
    
        pred_ksp       = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(weighted_cplx.unsqueeze(3)*sensitivity,dim=(0,1,2)), dim=(0,1,2)), dim=(0, 1,2))   
        
        #for INR network, it's better to use single FA to update the parameters
        idx            = torch.randint(low=0,high=8,size=(1,))
        pred_ksp_batch = pred_ksp.select(-1,idx.item())
        ksp_batch      = kspace.select(-1,idx.item())
        weighted_cplx_batch = weighted_cplx.select(-1,idx.item())
    
        mae_loss      = MAE_loss_function(torch.view_as_real(pred_ksp_batch.masked_select(ksp_batch!=0)).float(), 
                                                torch.view_as_real(ksp_batch.masked_select(ksp_batch!=0)).float())
        
        tv_loss       = TV_loss_function(weighted_cplx_batch.real) + TV_loss_function(weighted_cplx_batch.imag)
    
        loss1  = mae_loss + 0.01*tv_loss  # cooresponding to the loss1 in the paper
    
        pred_ksp_dc   = pred_ksp*(1-mask) + kspace*mask     #[FE, PE1, PE2, Coil, FA]
        pred_weighted_dc = torch.sum(torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(pred_ksp_dc,dim=(0,1,2)), dim=(0,1,2)), dim=(0, 1,2))*torch.conj(sensitivity),dim=3)   
        pred_weighted_sos = torch.sum(torch.abs(torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(pred_ksp_dc,dim=(0,1,2)), dim=(0,1,2)), dim=(0, 1,2)))**2,dim=3)**0.5   
    #    
        img_initial    = torch.abs(torch.stack(
                Utils.BTS_signal(TR, TR_BTS,EW, EW0.permute(3,0,1,2), T1R, initial_T1, initial_I0, initial_f, initial_k, flip_angles.permute(3,0,1,2)
            ), dim=-1) )      #[FE, PE1, PE2, nFA]
        
        img_initial_batch = img_initial.select(-1,idx.item())
        pred_weighted_sos_batch = pred_weighted_sos.abs().select(-1,idx.item())
    
        loss2  = MAE_loss_function(img_initial_batch.float(), pred_weighted_sos_batch.float())  # cooresponding to the loss2 in the paper
        
        if (e+1) < 1501:
            loss = loss1 + loss2 
            I0_list = [initial_I0,initial_I0]
            T1_list = [initial_T1,initial_T1]
            f_list  = [initial_f,initial_f]
            k_list  = [initial_k,initial_k]
            img_list = [img_initial,img_initial]
            loss3 = torch.tensor(0.0, requires_grad=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            writer.add_scalar('Loss/loss1 + loss2', loss1.item() + loss2.item(), e)
            writer.add_scalar('Loss/loss3', loss3, e)
         
        
        else:
            if (e+1) == 1501:
                for param in INR_init_model.Hash_enc3.parameters():
                    param.requires_grad = False

            '''This is proximal gradient physical module'''
            I0_list, T1_list, f_list, k_list, img_list = Physics_refine_model(mask, pred_weighted_dc.angle().detach(),kspace,sensitivity, flip_angles.permute(3,0,1,2), TR, TR_BTS, EW, EW0.permute(3,0,1,2), T1R, initial_T1, initial_I0, initial_f, initial_k)
    
            loss3 = compute_cost_ksp(img_list, kspace, sensitivity)
    
            loss = loss1 + loss2 + loss3 
    
            writer.add_scalar('Loss/loss1 + loss2 + loss3', loss1.item() + loss2.item() + loss3.item(), e)
            writer.add_scalar('Loss/loss1 + loss2', loss1.item() + loss2.item(), e)
            writer.add_scalar('Loss/loss3', loss3, e)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        loss_train += loss.item()
        iter_loop.set_postfix(loss='{:.4f}'.format(loss_train/(e+1)), 
                              loss1='{:.4f}'.format(loss1.item()), 
                              loss2='{:.4f}'.format(loss2.item()),
                              loss3='{:.4f}'.format(loss3.item()))
                           
        if (e+1)%1500 == 0:
            with torch.no_grad():
                sio.savemat(os.path.join(result_dir,f'pred_results_{e+1}.mat'),      {'pred_img_cplx':    weighted_cplx.cpu().detach().numpy(),
                                                                                      'pred_img_abs':     weighted_cplx.abs().cpu().detach().numpy(),
                                                                                      'pred_img_angle':   weighted_cplx.angle().cpu().detach().numpy(),
                                                                                      'pred_weighted_dc': pred_weighted_dc.cpu().detach().numpy(),
                                                                                      'initial_I0_abs':   initial_I0.cpu().detach().numpy(),
                                                                                      'initial_T1':       initial_T1.cpu().detach().numpy(),
                                                                                      'initial_f':        initial_f.cpu().detach().numpy(),
                                                                                      'initial_k':        initial_k.cpu().detach().numpy(),
                                                                                      'I0_list':          torch.stack(I0_list,-1).cpu().detach().numpy(),
                                                                                      'T1_list':          torch.stack(T1_list,-1).cpu().detach().numpy(),
                                                                                      'f_list':           torch.stack(f_list,-1).cpu().detach().numpy(),
                                                                                      'k_list':           torch.stack(k_list,-1).cpu().detach().numpy(),
                                                                                      'kf_list':           (torch.stack(k_list,-1).cpu().detach().numpy())*(torch.stack(f_list,-1).cpu().detach().numpy()),
                                                                                      'img_list':         torch.stack(img_list,-1).cpu().detach().numpy(),
                                                                                      })
                
                # save the model
                torch.save(INR_init_model.state_dict(), os.path.join(model_dir, f'MLPnet1_{e+1}.pkl'))
                torch.save(Physics_refine_model.state_dict(), os.path.join(model_dir, f'MLPnet2_{e+1}.pkl'))

          





    





