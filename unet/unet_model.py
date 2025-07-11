""" Full assembly of the parts to form the complete network """
import torch

from .unet_parts import *
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.bilinear     = bilinear

        self.inc          = (DoubleConv(in_channels, 32))
        self.down1        = (Down(32, 64))
        self.down2        = (Down(64, 128))
        self.down3        = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down4        = (Down(256, 512 // factor))
        self.up1          = (Up(512, 256 // factor, bilinear))
        self.up2          = (Up(256, 128 // factor, bilinear))
        self.up3          = (Up(128, 64 // factor, bilinear))
        self.up4          = (Up(64, 32, bilinear))
        self.outc         = (OutConv(32, out_channels))


    def forward(self, x0, return_feature = False):
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        doecoder_x1 = self.up1(x5, x4)
        doecoder_x2 = self.up2(doecoder_x1, x3)
        doecoder_x3 = self.up3(doecoder_x2, x2)
        doecoder_x4 = self.up4(doecoder_x3, x1)
        x           = self.outc(doecoder_x4)
        outx = x + x0

        if return_feature:
            return outx, x5, doecoder_x1, doecoder_x2, doecoder_x3, doecoder_x4
        else:
            return outx

   
