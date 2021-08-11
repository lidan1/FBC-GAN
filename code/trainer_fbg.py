from __future__ import print_function
from six.moves import range
import sys
import numpy as np
import os
import random
import time
from PIL import Image
from copy import deepcopy

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
from torch.nn.functional import softmax, log_softmax
from torch.nn.functional import cosine_similarity
from tensorboardX import summary
from tensorboardX import FileWriter

from miscc.config import cfg
from miscc.utils import mkdir_p

# from model import G_NET, D_NET
from background import G_NET, D_NET
from foreground import SPADEGenerator, NLayerDiscriminator, OutputFG
import models.networks as networks


# ################## Shared functions ###################

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.orthogonal(m.weight.data, 1.0)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('Linear') != -1:
		nn.init.orthogonal(m.weight.data, 1.0)
		if m.bias is not None:
			m.bias.data.fill_(0.0)


def load_params(model, new_param):
	for p, new_p in zip(model.parameters(), new_param):
		p.data.copy_(new_p)


def copy_G_params(model):
	flatten = deepcopy(list(p.data for p in model.parameters()))
	return flatten

def load_network(gpus):
	Gb1 = G_NET()
	Gb1.apply(weights_init)
	Gb1 = torch.nn.DataParallel(Gb1, device_ids=gpus)
	print(Gb1)

	Gf1 = SPADEGenerator()
	Gf1.init_weights('normal', gain=0.02)
	Gf1 = torch.nn.DataParallel(Gf1, device_ids=gpus)
	print(Gf1)

	G_out = OutputFG()
	G_out.init_weights('normal', gain=0.02)
	G_out = torch.nn.DataParallel(G_out, device_ids=gpus)
	print(G_out)

	D_b = D_NET()
	D_b.apply(weights_init)
	D_b = torch.nn.DataParallel(D_b, device_ids=gpus)

	D_f = NLayerDiscriminator()
	D_f.init_weights('normal', gain=0.02)
	D_f = torch.nn.DataParallel(D_f, device_ids=gpus)

	count = 0
	if cfg.TRAIN.GB1 != '':
		state_dict = torch.load(cfg.TRAIN.GB1)
		Gb1.load_state_dict(state_dict)
		print('Load ', cfg.TRAIN.GB1)

		istart = cfg.TRAIN.GB1.rfind('_') + 1
		iend = cfg.TRAIN.GB1.rfind('.')
		count = cfg.TRAIN.GB1[istart:iend]
		count = int(count) + 1

	if cfg.TRAIN.GF1 != '':
		state_dict = torch.load(cfg.TRAIN.GF1)
		Gf1.load_state_dict(state_dict)
		print('Load ', cfg.TRAIN.GF1)

	if cfg.TRAIN.G_OUT != '':
		state_dict = torch.load(cfg.TRAIN.G_OUT)
		G_out.load_state_dict(state_dict)
		print('Load ', cfg.TRAIN.G_OUT)

	if cfg.TRAIN.DB != '':
		print('Load %s.pth' % (cfg.TRAIN.DB))
		state_dict = torch.load('%s.pth' % (cfg.TRAIN.DB))
		D_b.load_state_dict(state_dict)

	if cfg.TRAIN.DF != '':
		print('Load %s.pth' % (cfg.TRAIN.DF))
		state_dict = torch.load('%s.pth' % (cfg.TRAIN.DF))
		D_f.load_state_dict(state_dict)


	if cfg.CUDA:
		Gb1.cuda()
		Gf1.cuda()
		G_out.cuda()
		D_b.cuda()
		D_f.cuda()

	return Gb1, Gf1, G_out, D_b, D_f, count


def define_optimizers(Gf1, Gb1, G_out, D_f, D_b):
	G_paras = list(Gf1.parameters()) + list(Gb1.parameters()) + list(G_out.parameters())
	optimizerG = torch.optim.Adam(G_paras,
						lr=cfg.TRAIN.GENERATOR_LR,
						betas=(0.5, 0.999))

	optimizerDf = torch.optim.Adam(D_f.parameters(),
						lr=cfg.TRAIN.DISCRIMINATOR_LR,
						betas=(0.5, 0.999))

	optimizerDb = torch.optim.Adam(D_b.parameters(),
						lr=cfg.TRAIN.DISCRIMINATOR_LR,
						betas=(0.5, 0.999))

	return optimizerG, optimizerDf, optimizerDb


# def save_model(netG, avg_param_G, netD, epoch, model_dir):
# 	load_params(netG, avg_param_G)
# 	torch.save(
# 		netG.state_dict(),
# 		'%s/netG_%d.pth' % (model_dir, epoch))
# 	torch.save(
# 		netD.state_dict(),
# 		'%s/netD%d.pth' % (model_dir, epoch))
# 	print('Save G/Ds models.')

def save_model(Gf1, Gb1, G_out, D_f, D_b, epoch, model_dir):
	# load_params(netG, avg_param_G)
	torch.save(
		Gf1.state_dict(),
		'%s/netGf1_%d.pth' % (model_dir, epoch))
	torch.save(
		Gb1.state_dict(),
		'%s/netGf1_%d.pth' % (model_dir, epoch))
	torch.save(
		G_out.state_dict(),
		'%s/netGout_%d.pth' % (model_dir, epoch))
	torch.save(
		D_f.state_dict(),
		'%s/netDf%d.pth' % (model_dir, epoch))
	torch.save(
		D_b.state_dict(),
		'%s/netDb%d.pth' % (model_dir, epoch))
	print('Save G/Ds models.')


def save_img_results(imgs_tcpu, fake_imgs, num_imgs,
					 count, image_dir, summary_writer):
	# num = cfg.TRAIN.VIS_COUNT
	num = 5
	real_img = imgs_tcpu[-1][0:num]
	vutils.save_image(
		real_img, '%s/real_samples%09d.png' % (image_dir,count),
		normalize=True)
	real_img_set = vutils.make_grid(real_img).numpy()
	real_img_set = np.transpose(real_img_set, (1, 2, 0))
	real_img_set = real_img_set * 255
	real_img_set = real_img_set.astype(np.uint8)

	for i in range(len(fake_imgs)):
		fake_img = fake_imgs[i][0:num]

		vutils.save_image(
			fake_img.data, '%s/count_%09d_fake_samples%d.png' %
			(image_dir, count, i), normalize=True)
		fake_img_set = vutils.make_grid(fake_img.data).cpu().numpy()
		fake_img_set = np.transpose(fake_img_set, (1, 2, 0))
		fake_img_set = (fake_img_set + 1) * 255 / 2
		fake_img_set = fake_img_set.astype(np.uint8)

		summary_writer.flush()



class FBG_trainer(object):
	def __init__(self, output_dir, data_loader, imsize):
		if cfg.TRAIN.FLAG:
			self.model_dir = os.path.join(output_dir, 'Model')
			self.image_dir = os.path.join(output_dir, 'Image')
			self.log_dir = os.path.join(output_dir, 'Log')
			mkdir_p(self.model_dir)
			mkdir_p(self.image_dir)
			mkdir_p(self.log_dir)
			self.summary_writer = FileWriter(self.log_dir)

		s_gpus = cfg.GPU_ID.split(',')
		self.gpus = [int(ix) for ix in s_gpus]
		self.num_gpus = len(self.gpus)
		torch.cuda.set_device(self.gpus[0])
		cudnn.benchmark = True

		self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
		self.max_epoch = cfg.TRAIN.MAX_EPOCH
		self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

		self.data_loader = data_loader
		self.num_batches = len(self.data_loader)

	def prepare_data(self, data):
		#  train return fimgs, cimgs, cmasks, key, warped_bbox 
		# test return imgs, masks, key 

		fimgs, cimgs, cmasks, _, warped_bbox = data
		real_vfimgs, real_vcimgs, real_vcmasks = [], [], []
		if cfg.CUDA:
			for i in range(len(warped_bbox)):
				warped_bbox[i] = Variable(warped_bbox[i]).float().cuda()
		else:
			for i in range(len(warped_bbox)):
				warped_bbox[i] = Variable(warped_bbox[i])

		if cfg.CUDA:
			real_vfimgs.append(Variable(fimgs[0]).cuda())
			real_vcimgs.append(Variable(cimgs[0]).cuda())
			real_vcmasks.append(Variable(cmasks[0]).cuda())

		else:
			real_vfimgs.append(Variable(fimgs[0]))
			real_vcimgs.append(Variable(cimgs[0]))
			real_vcmasks.append(Variable(cmasks[0]))

		return fimgs, real_vfimgs, real_vcimgs, real_vcmasks, warped_bbox

	def train_Db(self, count):
		flag = count % 100
		batch_size = self.real_fimgs[0].size(0)
		criterion, criterion_one = self.criterion, self.criterion_one
		# netD, optD = self.netD[idx], self.optimizersD[idx] 
		netD, optD = self.D_b, self.optimizerDb
		optD.zero_grad()
		real_imgs = self.real_fimgs[0]
		fake_imgs = self.bg[0]
		real_logits = netD(real_imgs)

		fake_labels = torch.zeros_like(real_logits[1])
		ext, output = real_logits
		weights_real = torch.ones_like(output)
		real_labels = torch.ones_like(output)
				
		for i in range(batch_size):
			x1 =  self.warped_bbox[0][i]
			x2 =  self.warped_bbox[2][i]
			y1 =  self.warped_bbox[1][i]
			y2 =  self.warped_bbox[3][i]

			a1 = max(torch.tensor(0).float().cuda(), torch.ceil((x1 - self.recp_field)/self.patch_stride))
			a2 = min(torch.tensor(self.n_out - 1).float().cuda(), torch.floor((self.n_out - 1) - ((126 - self.recp_field) - x2)/self.patch_stride)) + 1
			b1 = max(torch.tensor(0).float().cuda(), torch.ceil((y1 - self.recp_field)/self.patch_stride))
			b2 = min(torch.tensor(self.n_out - 1).float().cuda(), torch.floor((self.n_out - 1) - ((126 - self.recp_field) - y2)/self.patch_stride)) + 1

			if (x1 != x2 and y1 != y2):
				weights_real[i, :, a1.type(torch.int) : a2.type(torch.int) , b1.type(torch.int) : b2.type(torch.int)] = 0.0

				norm_fact_real = weights_real.sum()
				norm_fact_fake = weights_real.shape[0]*weights_real.shape[1]*weights_real.shape[2]*weights_real.shape[3]
				real_logits = ext, output

				fake_logits = netD(fake_imgs.detach())
		
		# Background stage

		errD_real_uncond = criterion(real_logits[1], real_labels)  # Real/Fake loss for 'real background' (on patch level)
		errD_real_uncond = torch.mul(errD_real_uncond, weights_real)  # Masking output units which correspond to receptive fields which lie within the boundin box
		errD_real_uncond = errD_real_uncond.mean()

		errD_real_uncond_classi = criterion(real_logits[0], weights_real)  # Background/foreground classification loss
		errD_real_uncond_classi = errD_real_uncond_classi.mean()
	
		errD_fake_uncond = criterion(fake_logits[1], fake_labels)  # Real/Fake loss for 'fake background' (on patch level)
		errD_fake_uncond = errD_fake_uncond.mean()

		if (norm_fact_real > 0):    # Normalizing the real/fake loss for background after accounting the number of masked members in the output.
			errD_real = errD_real_uncond * ((norm_fact_fake * 1.0) /(norm_fact_real * 1.0))
		else:
			errD_real = errD_real_uncond

		errD_fake = errD_fake_uncond
		errD = ((errD_real + errD_fake) * cfg.TRAIN.BG_LOSS_WT) + errD_real_uncond_classi

		errD.backward()
		optD.step()

		if (flag == 0):
			summary_D = summary.scalar('Db_loss', errD.data)
			self.summary_writer.add_summary(summary_D, count)
			summary_D_real = summary.scalar('D_loss_real', errD_real.data)
			self.summary_writer.add_summary(summary_D_real, count)
			summary_D_fake = summary.scalar('D_loss_fake', errD_fake.data)
			self.summary_writer.add_summary(summary_D_fake, count)

		return errD

	# Take the prediction of fake and real images from the combined batch
	def divide_pred(self, pred):
		# the prediction contains the intermediate outputs of multiscale GAN,
		# so it's usually a list
		if type(pred) == list:
			fake = []
			real = []
			for p in pred:
				fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
				real.append([tensor[tensor.size(0) // 2:] for tensor in p])
		else:
			fake = pred[:pred.size(0) // 2]
			real = pred[pred.size(0) // 2:]

		return fake, real
		
	def train_Df(self, count):
		flag = count % 100
		batch_size = self.real_fimgs[0].size(0)
		criterionGAN = self.criterionGAN
		# netD, optD = self.netD[idx], self.optimizersD[idx] 
		netD, optD = self.D_f, self.optimizerDf
		optD.zero_grad()
		Df_losses = {}
		real_imgs = self.real_cimgs[0]
		fake_imgs = self.fg
		mask = self.real_cmasks[0]
		print('mask',mask.shape)
		print('img',fake_imgs.shape)
		fake_concat = torch.cat([mask, fake_imgs], dim=1)
		real_concat = torch.cat([mask, real_imgs], dim=1)

		fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
		discriminator_out = netD(fake_and_real)
		pred_fake, pred_real = self.divide_pred(discriminator_out)
		
		Df_losses['Df_Fake'] = criterionGAN(pred_fake, False,
											   for_discriminator=True)
		Df_losses['Df_real'] = criterionGAN(pred_real, True,
											   for_discriminator=True)
		errD = sum(Df_losses.values()).mean()
		errD.backward()
		optD.step()
		return Df_losses

	def train_G(self, count):
		flag = count % 100
		optG = self.optimizerG
		optG.zero_grad()
		G_losses = {}
		batch_size = self.real_fimgs[0].size(0)
		criterion_one, criterion_class = self.criterion_one, self.criterion_class
		criterionGAN = self.criterionGAN
		real_imgs = self.real_cimgs[0]
		fake_imgs = self.fg[0]
		mask = self.real_cmasks[0]

		# Db
		db_outputs = self.netDb(self.fake_imgs[0]) 	
		# real/fake loss for background (0) and child (2) stage
		real_labels = torch.ones_like(db_outputs[1])
		errG = criterion_one(db_outputs[1], real_labels) 
		errG = errG * cfg.TRAIN.BG_LOSS_WT
		errG_classi = criterion_one(db_outputs[0], real_labels) # Background/Foreground classification loss for the fake background image (on patch level)
		G_losses['Gb_adv'] = errG
		G_losses['Gb_aux'] = errG_classi

		# Df
		fake_concat = torch.cat([mask, fake_imgs], dim=1)
		real_concat = torch.cat([mask, real_imgs], dim=1)
		fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
		discriminator_out = self.netDf(fake_and_real)
		pred_fake, pred_real = self.divide_pred(discriminator_out)
		G_losses['Gf_adv'] = criterionGAN(pred_fake, True,
											for_discriminator=False)
		G_losses['Gf_vgg'] = self.criterionVGG(fake_imgs, real_imgs) \
				* self.opt.lambda_vgg

		num_D = len(pred_fake)
		lambda_feat = 10
		GAN_Feat_loss = self.FloatTensor(1).fill_(0)
		for i in range(num_D):  # for each discriminator
			# last output is the final prediction, so we exclude it
			num_intermediate_outputs = len(pred_fake[i]) - 1
			for j in range(num_intermediate_outputs):  # for each layer output
				unweighted_loss = self.criterionFeat(
					pred_fake[i][j], pred_real[i][j].detach())
				GAN_Feat_loss += unweighted_loss * lambda_feat / num_D
		G_losses['Gf_feat'] = GAN_Feat_loss
		
		errGb = G_losses['Gb_adv'] + G_losses['Gb_aux']
		errGf = G_losses['Gf_adv'] + G_losses['Gf_vgg'] + G_losses['Gf_fea']
		errG = errGb + errGf

		g_loss = sum(G_losses.values()).mean()
		g_loss.backward()
		self.optG.step()

		summary_D = summary.scalar('G_loss', errG.data)
		self.summary_writer.add_summary(summary_D, count) 

		return errG

	def train(self):
		self.Gb1, self.Gf1, self.G_out, self.D_b, self.D_f, start_count = load_network(self.gpus)
		avg_param_Gb1 = copy_G_params(self.Gb1)
		avg_param_Gf1 = copy_G_params(self.Gf1)
		avg_param_G_out = copy_G_params(self.G_out)

		self.optimizerG, self.optimizerDf, self.optimizerDb =\
			define_optimizers(self.Gf1, self.Gb1, self.G_out, self.D_f, self.D_b)

		self.criterion = nn.BCELoss(reduce=False)
		self.criterion_one = nn.BCELoss()
		self.criterion_class = nn.CrossEntropyLoss()
		self.criterionFeat = torch.nn.L1Loss()
		self.criterionVGG = networks.VGGLoss(self.gpus)
		self.FloatTensor = torch.cuda.FloatTensor if cfg.CUDA else torch.FloatTensor
		self.criterionGAN = networks.GANLoss(gan_mode = 'hinge', tensor=self.FloatTensor)

		self.real_labels = \
			Variable(torch.FloatTensor(self.batch_size).fill_(1))
		self.fake_labels = \
			Variable(torch.FloatTensor(self.batch_size).fill_(0))
	
		nz = cfg.GAN.Z_DIM
		noise = Variable(torch.FloatTensor(self.batch_size, nz))
		fixed_noise = \
			Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))
		# hard_noise = \
		# 	Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1)).cuda()
	
		self.patch_stride = float(4)    # Receptive field stride given the current discriminator architecture for background stage 
		self.n_out = 24                 # Output size of the discriminator at the background stage; N X N where N = 24
		self.recp_field = 34            # Receptive field of each of the member of N X N


		if cfg.CUDA:
			self.criterion.cuda()
			self.criterion_one.cuda()
			self.criterion_class.cuda()
			self.criterionFeat.cuda()
			self.criterionVGG.cuda()
			self.criterionGAN.cuda()

			self.real_labels = self.real_labels.cuda()
			self.fake_labels = self.fake_labels.cuda()
			noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
		
		print ("Starting normal FineGAN training..") 
		count = start_count
		start_epoch = start_count // (self.num_batches)

		for epoch in range(start_epoch, self.max_epoch):
			start_t = time.time()
			
			for step, data in enumerate(self.data_loader, 0):
				# fimgs, real_vfimgs, real_vcimgs, real_vcmasks, warped_bbox
				self.imgs_tcpu, self.real_fimgs, self.real_cimgs, self.real_cmasks, \
					self.warped_bbox = self.prepare_data(data)
				# print(self.real_cimgs)
				# print(step)
				print(self.real_cimgs[0].shape)
				print(self.real_cmasks[0].shape)

				# Feedforward through Generator. Obtain stagewise fake images
				z_f = torch.randn(self.batch_size, cfg.SPADE.Z_DIM,
						dtype=torch.float32, device=self.real_cimgs[0].get_device())
				z_b = torch.randn(self.batch_size, cfg.SPADE.Z_DIM,
						dtype=torch.float32, device=self.real_cimgs[0].get_device())
				
				print('device',self.real_cimgs[0].get_device())
				self.ff = self.Gf1(self.real_cmasks[0], z_f)
				self.fb = self.Gb1(z_b)
				self.fg, self.bg = self.G_out(self.ff, self.fb)  # generated foreground and background
				self.output = self.real_cmasks[0] * self.fg + (1-self.real_cmasks[0]) * self.bg # composit output img

				# Update Discriminator networks 
				# errD_f = 0
				errD_f = self.train_Df(count)
				# errD_f += errD

				errD_b = 0
				errD = self.train_Db(count)
				errD_b += errD

				errG = self.train_G(count)

				count = count + 1

				if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
					backup_para = copy_G_params(self.netG)
					save_model(self.Gf1, self.Gb1, self.G_out, self.D_f, self.D_b, count, self.model_dir)
					# save_model(self.netG, avg_param_G, self.netD, count, self.model_dir)
					# Save images
					# load_params(self.netG, avg_param_G)
					self.netG.eval()
					with torch.set_grad_enabled(False):
						z_f = torch.randn(self.batch_size, cfg.SPADE.Z_DIM,
								dtype=torch.float32, device=self.real_cimgs.get_device())
						z_b = torch.randn(self.batch_size, cfg.SPADE.Z_DIM,
								dtype=torch.float32, device=self.real_cimgs.get_device())
						self.ff = self.Gf1(self.real_cmasks, z_f)
						self.fb = self.Gb1(z_b)
						self.fg, self.bg = self.G_out(self.ff, self.fb)  # generated foreground and background
						self.output = self.real_cmasks * self.fg + (1-self.real_cmasks) * self.bg # composit output img

						save_img_results(self.imgs_tcpu, (self.output), 0,
											count, self.image_dir, self.summary_writer)
					print(self.image_dir)
					self.netG.train()
					load_params(self.netG, backup_para)

			end_t = time.time()
			print('''[%d/%d][%d]
						 Loss_D: %.2f Loss_G: %.2f Time: %.2fs
					  '''  
				  % (epoch, self.max_epoch, self.num_batches,
					 errD_b.data, errG.data,
					 end_t - start_t))

		# save_model(self.netG, avg_param_G, self.netD, count, self.model_dir)
		save_model(self.Gf1, self.Gb1, self.G_out, self.D_f, self.D_b, count, self.model_dir)

		print ("Done with the normal training.")
		self.summary_writer.close()


