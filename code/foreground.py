"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm

from miscc.config import cfg
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.normalization import SPADE

import numpy as np
import util.util as util

# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, cfg):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        # if 'spectral' in opt.norm_G:
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = 'spadeinstance3x3'
        self.norm_0 = SPADE(spade_config_str, fin, cfg.SPADE.L_NC)
        self.norm_1 = SPADE(spade_config_str, fmiddle, cfg.SPADE.L_NC)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, cfg.SPADE.L_NC)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class AdaIN(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, y):
		eps = 1e-5	
		mean_x = torch.mean(x, dim=[2,3])
		mean_y = torch.mean(y, dim=[2,3])

		std_x = torch.std(x, dim=[2,3])
		std_y = torch.std(y, dim=[2,3])

		mean_x = mean_x.unsqueeze(-1).unsqueeze(-1)
		mean_y = mean_y.unsqueeze(-1).unsqueeze(-1)

		std_x = std_x.unsqueeze(-1).unsqueeze(-1) + eps
		std_y = std_y.unsqueeze(-1).unsqueeze(-1) + eps

		out = (x - mean_x)/ std_x * std_y + mean_y

		return out



class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self):
        super().__init__()
        self.opt = cfg
        nf = cfg.SPADE.NGF # 64

        self.sw, self.sh = 4, 4

        self.fc = nn.Linear(cfg.SPADE.Z_DIM, 16 * nf * self.sw * self.sh) # 16x64x4x4 = 16384
        # else:
        #     # Otherwise, we make the network deterministic by starting with
        #     # downsampled segmentation map instead of random z
        #     self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, cfg) # 4x4x64 1024

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, cfg) # 4x4x64 1024
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, cfg) # 4x4x64 1024

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, cfg) # 4x4x64 1024 512
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, cfg) # 512 256
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, cfg) # 256 128
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, cfg) # 128 64

        final_nc = nf

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, input, z=None):
        seg = input

        # if self.opt.use_vae:
        # we sample z from unit normal and reshape the tensor
        if z is None:
            z = torch.randn(input.size(0), cfg.SPADE.Z_DIM,
                            dtype=torch.float32, device=input.get_device())
        x = self.fc(z)
        x = x.view(-1, 16 * cfg.SPADE.NGF, self.sh, self.sw) # 1 x (1024x4x4)
        # else:
        #     # we downsample segmap and run convolution
        #     x = F.interpolate(seg, size=(self.sh, self.sw))
        #     x = self.fc(x)

        x = self.head_0(x, seg) #1024x4x4
        x = self.up(x)          #1024x8x8
        x = self.G_middle_0(x, seg) #1024x8x8
        x = self.G_middle_1(x, seg) #1024x8x8
        x = self.up(x)              #1024x16x16
        x = self.up_0(x, seg)       #512x16x16
        x = self.up(x)              #512x32x32
        x = self.up_1(x, seg)       #256x32x32
        x = self.up(x)              #256x64x64
        x = self.up_2(x, seg)       #128x64x64
        x = self.up(x)              #128x128x128
        x = self.up_3(x, seg)       #64x128x128
        # x = self.conv_img(F.leaky_relu(x, 2e-1)) #3x128x128
        # x = F.tanh(x)
        return x

class OutputFG(BaseNetwork):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),   # 64x128x128
                # nn.BatchNorm2d(64),
                nn.LeakyReLU(2e-1))

        self.adaIN = AdaIN()

        self.conv_img_f = nn.Sequential(
                    nn.Conv2d(64, 3, 3, padding=1),  # 3x128x128	
                    nn.Tanh())

        self.conv_img_b = nn.Sequential(
                    nn.Conv2d(64, 3, 3, padding=1),	 # 3x128x128		
                    nn.Tanh())        

    def forward(self, gf1, gb1):
        gf_ada = self.adaIN(gf1, gb1)
        alpha = 0.2
        gf1 = alpha * gf_ada + (1 - alpha) * gf1

        gf1 = self.base(gf1)
        gb1 = self.base(gb1)
        fg = self.conv_img_f(gf1)
        bg = self.conv_img_b(gb1)
        return fg, bg
      

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self):
        super().__init__()

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = cfg.SPADE.NDF
        input_nc = self.compute_D_input_nc(cfg)

        norm_layer = get_nonspade_norm_layer(cfg, norm_type='spectralinstance')
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, 4): # D_nlayers = 4
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == 4 - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, cfg):
        input_nc = cfg.SPADE.L_NC + cfg.SPADE.O_NC
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = True # not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
