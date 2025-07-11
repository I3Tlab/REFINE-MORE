import torch
from torch.utils.tensorboard import SummaryWriter
import scipy.io as sio
import mat73
import tinycudann as tcnn
import os
import loss_function as lf
import sys
import shutil
from model import REFINE_MORE
import Utils
from tqdm import tqdm

'''Parameter Setup'''
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
seedd  = 3598
torch.manual_seed(seedd) # seed the RNG for all devices (both CPU and CUDA)

phase_num        = 4
learning_rate    = 0.001
epoch            = 3000
data_dir         = './data'
out_dir          = './outputs'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model_dir  = os.path.join(out_dir, 'model')
log_dir    = os.path.join(out_dir, 'log')
result_dir = os.path.join(out_dir, 'result')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

try:
    current_file = os.path.abspath(sys.argv[0])
    shutil.copy2(current_file, out_dir)
    print(f"file saved: {out_dir}")
except Exception as e:
    print(f"save filed: {str(e)}")

writer = SummaryWriter(log_dir)

'''Load Data'''
kspace = mat73.loadmat(os.path.join(data_dir, f'kspace.mat'))[f'kspace']        # [FE, PE1, PE2, Coil, FA]
kspace = torch.complex(torch.Tensor(kspace.real), torch.Tensor(kspace.imag)).to(device)                                                                                    # [FE, PE1, PE2, Coil, FA]
print('Load kspace', kspace.shape, kspace.dtype)
nFE, nPE1, nPE2, nCoil, nFA = kspace.shape

### load the pre-calculated sensitivity maps
sensitivity = mat73.loadmat(os.path.join(data_dir, f'sensitivity_maps.mat'))[f'bart_sensitivity']  # [FE, PE1, PE2, Coil]
sensitivity = torch.complex(
    torch.Tensor(sensitivity.real), torch.Tensor(sensitivity.imag)
).to(device).unsqueeze(-1)
print('calculte sensitivity maps', sensitivity.shape, sensitivity.dtype)

tissue_mask = torch.ones((nFE, nPE1, nPE2))
temp        = sensitivity.sum(dim=3).sum(dim=3).abs()
tissue_mask[temp==0] = 0
tissue_mask = tissue_mask.to(device)
sio.savemat(os.path.join(result_dir, 'tissue_mask.mat'), {'tissue_mask':tissue_mask.cpu().detach().numpy()})

### fully sampled rss
images = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(kspace,dim=(0,1,2)), dim=(0,1,2)), dim=(0, 1,2))                                                   # [FE, PE1, PE2, Coil, FA]
fs_cmb = (images*torch.conj(sensitivity)).sum(dim=3)                                             # [FE, PE1, PE2, FA]
print('fs_cmb', fs_cmb.shape, fs_cmb.dtype)
fs_cmb = fs_cmb / torch.max(torch.abs(fs_cmb))
sio.savemat(os.path.join(result_dir, 'fs_cmb.mat'), {'fs_cmb':fs_cmb.cpu().detach().numpy()})


mask = sio.loadmat(os.path.join(data_dir,'Gaussian_2PE_mask.mat'))['mask']  #[PE1, PE2, FA]
mask = torch.Tensor(mask).to(device).unsqueeze(0).unsqueeze(3)
print('Load mask', mask.shape, mask.dtype)

### undersampleing
kspace = kspace * mask                                                                         # [FE, PE1, PE2, Coil, FA]

### calculate zero fiiling results
images = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(kspace,dim=(0,1,2)), dim=(0,1,2)), dim=(0, 1,2))   
rss    = torch.sqrt((images.abs()**2).sum(dim=3))                                           # [FE, PE1, PE2, FA]
kspace = kspace / torch.max(rss)  
print('rss', rss.shape, rss.dtype)
sio.savemat(os.path.join(result_dir, 'zf_rss.mat'), {'rss':rss.cpu().detach().numpy()})

### Load B1+
B1 = sio.loadmat(os.path.join(data_dir, f'B1.mat'))['B1']                                                      #  [FE, PE1, PE2]
B1 = torch.Tensor(B1).to(device)                                                                                 # [FE, PE1, PE2]
print('Load B1+', B1.shape, B1.dtype)

### flip_angles
a1 = torch.Tensor( 5.0*B1).to(device)
a2 = torch.Tensor(10.0*B1).to(device)
a3 = torch.Tensor(20.0*B1).to(device)
a4 = torch.Tensor(40.0*B1).to(device)
flip_angles = torch.stack([a1, a2, a3, a4], dim=-1)                                                             # [FE, PE1, PE2, FA]
print('flip_angles', flip_angles.shape, flip_angles.dtype)

### Load W0, W
BTS_sat = mat73.loadmat(os.path.join(data_dir, f'BTS_sat.mat'))
W0 = BTS_sat['w0']                                                                                          # [FE, PE1, PE2, FA]
W = BTS_sat['w']                                                                                            # [FE, PE1, PE2, FA]
W0 = torch.Tensor(W0).to(device) 
W = torch.Tensor(W).to(device)                                                                              # [FE, PE1, PE2]
print('W0', W0.shape, W0.dtype)
print('W', W.shape, W.dtype)

### pw, pw_bts, EW0, T1R
pw     = 0.0005 # [s]
pw_bts = 0.008 # [s]
EW0    = torch.exp(-1.0 * W0 * pw) # [FE, PE1, PE2, FA]
EW     = torch.exp(-1.0 * W * pw_bts) # [FE, PE1, PE2, FA]
T1R = torch.ones(EW.shape[0], EW.shape[1], EW.shape[2]).to(device)  # [FE, PE1, PE2]
print('EW0', EW0.shape, EW0.dtype)
print('EW', EW.shape, EW.dtype)
print('T1R', T1R.shape, T1R.dtype)

TR = 0.04
TR_BTS = 0.04

REFINE_MORE(kspace.real, kspace.imag, sensitivity.real,sensitivity.imag,mask,tissue_mask,TR,TR_BTS,EW,EW0,T1R,flip_angles,seedd,learning_rate,phase_num,epoch,out_dir,log_dir,model_dir,result_dir,device)

