import sys
import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Upsample


class GLU(nn.Module):
	def __init__(self):
		super(GLU, self).__init__()

	def forward(self, x):
		nc = x.size(1)
		assert nc % 2 == 0, 'channels dont divide 2!'
		nc = int(nc/2)
		return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
	"3x3 convolution with padding"
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
					 padding=1, bias=False)


def convlxl(in_planes, out_planes):
	"3x3 convolution with padding"
	return nn.Conv2d(in_planes, out_planes, kernel_size=13, stride=1,
					 padding=1, bias=False)


# ############## G networks ################################################
# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
	block = nn.Sequential(
		nn.Upsample(scale_factor=2, mode='nearest'),
		conv3x3(in_planes, out_planes * 2),
		nn.BatchNorm2d(out_planes * 2),
		GLU()
	)
	return block

def upBlock3x3_leakRelu(in_planes, out_planes):
	block = nn.Sequential(
		nn.Upsample(scale_factor=2, mode='nearest'),
		conv3x3(in_planes, out_planes),
		nn.BatchNorm2d(out_planes),
		nn.LeakyReLU(0.2, inplace=True)
	)
	return block

def sameBlock(in_planes, out_planes):
	block = nn.Sequential(
		conv3x3(in_planes, out_planes * 2),
		nn.BatchNorm2d(out_planes * 2),
		GLU()
	)
	return block

# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
	block = nn.Sequential(
		conv3x3(in_planes, out_planes),
		nn.BatchNorm2d(out_planes),
		nn.ReLU()
	)
	return block

class ResBlock(nn.Module):
	def __init__(self, channel_num):
		super(ResBlock, self).__init__()
		self.block = nn.Sequential(
			conv3x3(channel_num, channel_num * 2),
			nn.BatchNorm2d(channel_num * 2),
			GLU(),
			conv3x3(channel_num, channel_num),
			nn.BatchNorm2d(channel_num)
		)

	def forward(self, x):
		residual = x
		out = self.block(x)
		out += residual
		return out

class G_NET(nn.Module):
	def __init__(self):
		super(G_NET, self).__init__()
		self.gf_dim = cfg.GAN.GF_DIM
		self.upsampling = Upsample(scale_factor = 2, mode = 'bilinear')
		self.scale_fimg = nn.UpsamplingBilinear2d(size = [126, 126])
		
		ngf = self.gf_dim
		self.fc = nn.Sequential(
			nn.Linear(cfg.SPADE.Z_DIM, 16 * ngf * 4 * 4, bias=False),
			nn.BatchNorm1d(16 * ngf * 4 * 4))

		self.upsample1 = upBlock3x3_leakRelu(16 * ngf, 8 * ngf)
		self.upsample2 = upBlock3x3_leakRelu(8 * ngf, 4 * ngf)
		self.upsample3 = upBlock3x3_leakRelu(4 * ngf, 2 * ngf)
		self.upsample4 = upBlock3x3_leakRelu(2 * ngf, ngf)
		self.upsample5 = upBlock3x3_leakRelu(ngf, ngf)

	def forward(self, z_code):

		# in_code = torch.cat((code, z_code), 1)
		in_code = z_code
		out_code = self.fc(in_code)
		out_code = out_code.view(-1, 16 * self.gf_dim, 4, 4) # 1024x4x4
		out_code = self.upsample1(out_code)					# 512x8x8
		out_code = self.upsample2(out_code)					# 256x16x16
		out_code = self.upsample3(out_code)					# 128x32x32	
		out_code = self.upsample4(out_code)					# 64x64x64
		out_code = self.upsample5(out_code)					# 64x128x128

		return out_code

# ############## D networks ################################################
def Block3x3_leakRelu(in_planes, out_planes):
	block = nn.Sequential(
		conv3x3(in_planes, out_planes),
		nn.BatchNorm2d(out_planes),
		nn.LeakyReLU(0.2, inplace=True)
	)
	return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
	block = nn.Sequential(
		nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
		nn.BatchNorm2d(out_planes),
		nn.LeakyReLU(0.2, inplace=True)
	)
	return block

def encode_background_img(ndf): # Defines the encoder network used for background image
	encode_img = nn.Sequential(
		nn.Conv2d(3, ndf, 4, 2, 0, bias=False),
		nn.LeakyReLU(0.2, inplace=True),
		nn.Conv2d(ndf, ndf * 2, 4, 2, 0, bias=False),
		nn.LeakyReLU(0.2, inplace=True),
		nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 0, bias=False),
		nn.LeakyReLU(0.2, inplace=True),
	)
	return encode_img


class D_NET(nn.Module):
	def __init__(self, ):
		super(D_NET, self).__init__()
		self.df_dim = cfg.GAN.DF_DIM
		self.ef_dim = 1
		self.define_module()

	def define_module(self):
		ndf = self.df_dim
		efg = self.ef_dim

		self.patchgan_img_code_s16 = encode_background_img(ndf)
		self.uncond_logits1 = nn.Sequential(
		nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1),
		nn.Sigmoid())
		self.uncond_logits2 = nn.Sequential(
		nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1),
		nn.Sigmoid())

	def forward(self, x_var):

		x_code = self.patchgan_img_code_s16(x_var)
		classi_score = self.uncond_logits1(x_code) # Background vs Foreground classification score (0 - background and 1 - foreground) 
		rf_score = self.uncond_logits2(x_code) # Real/Fake score for the background image
		return [classi_score, rf_score]



