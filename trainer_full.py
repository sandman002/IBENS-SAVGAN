
import os
import time
import torch
import datetime
import random
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

# from myDis import Generator
# from myDis import Discriminator
import timeit
# from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import socket

from model_full import Generator
from model_full import Discriminator
from model_full import Discriminator2

# from myModel_wNyattn_up import Generator
# from myModel_wNyattn_up import Discriminator

from utils import *
from helper_plot import plot_attn
from makeminimontage import makeminimontage
from skimage.external.tifffile import TiffWriter
Tensor = torch.cuda.FloatTensor
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
# save_dir = './run/'
# log_dir = os.path.join(save_dir, '2stream', datetime.datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
writer = SummaryWriter()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class Trainer(object):
	def __init__(self, data_loader, config):

		# Data loader
		self.data_loader = data_loader

		# exact model and loss
		self.model = config.model
		self.adv_loss = config.adv_loss

		# Model hyper-parameters
		self.imsize = config.imsize
		self.frames = config.frames
		self.g_num = config.g_num
		self.z_dim = config.z_dim
		self.g_conv_dim = config.g_conv_dim
		self.d_conv_dim = config.d_conv_dim
		self.parallel = config.parallel

		self.lambda_gp = config.lambda_gp
		self.total_step = config.total_step
		self.d_iters = config.d_iters
		self.batch_size = config.batch_size
		self.num_workers = config.num_workers
		self.g_lr = config.g_lr
		self.d_lr = config.d_lr
		self.lr_decay = config.lr_decay
		self.beta1 = config.beta1
		self.beta2 = config.beta2
		self.pretrained_model = config.pretrained_model

		self.dataset = config.dataset
		self.use_tensorboard = config.use_tensorboard
		self.image_path = config.image_path
		self.log_path = config.log_path
		self.model_save_path = config.model_save_path
		self.sample_path = config.sample_path
		self.log_step = config.log_step
		self.sample_step = config.sample_step
		self.model_save_step = config.model_save_step
		self.version = config.version
		self.gimlosswt = config.gimlosswt

		

		# Path
		self.log_path = os.path.join(config.log_path, self.version)
		self.sample_path = os.path.join(config.sample_path, self.version)
		self.model_save_path = os.path.join(config.model_save_path, self.version)

		self.build_model()

		# if self.use_tensorboard:
		# 	self.build_tensorboard()

		# Start with trained model
		if self.pretrained_model:
			self.load_pretrained_model()

	def train(self):
		# Data iterator
		input_A = Tensor(self.batch_size, 1, self.frames, 64, 64)
		
		data_iter = iter(self.data_loader)
		step_per_epoch = len(self.data_loader)
		print("step_per_epoch", step_per_epoch)
		model_save_step = 500

		# Fixed input for debugging
		fixed_z = tensor2var(torch.randn(4, self.z_dim))

		# Start with trained model
		if self.pretrained_model:
			start = self.pretrained_model + 1
		else:
			start = 0

		# Start time
		start_time = time.time()
		for step in range(start, self.total_step):

			# ================== Train D ================== #
			self.D.train()
			self.Dim.train()
			self.G.train()

			for i in range(0,self.d_iters):
				data_iter = iter(self.data_loader)
				real_vid = input_A.copy_(next(data_iter)[0])
				
				
				real_vid = tensor2var(real_vid)
				# print("read vid", real_vid.shape)
				d_out_real = self.D(real_vid)
			
				
				if self.adv_loss == 'wgan-gp':
					d_loss_real = - torch.mean(d_out_real)
				elif self.adv_loss == 'hinge':
					d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

				# apply Gumbel Softmax
				z = tensor2var(torch.randn(real_vid.size(0), self.z_dim))
				fake_images = self.G(z)
				
				
				d_out_fake = self.D(fake_images)
				if self.adv_loss == 'wgan-gp':
					d_loss_fake = d_out_fake.mean()
				elif self.adv_loss == 'hinge':
					d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()


				# Backward + Optimize
				d_loss = d_loss_real + d_loss_fake
				self.reset_grad()
				d_loss.backward()
				self.d_optimizer.step()


				if self.adv_loss == 'wgan-gp':
					# Compute gradient penalty
					alpha = torch.rand(real_vid.size(0), 1, 1, 1, 1).cuda().expand_as(real_vid)
					# print("sizes: ", real_vid.shape, fake_images.shape)
					interpolated = Variable(alpha * real_vid.data + (1 - alpha) * fake_images.data, requires_grad=True)
					out = self.D(interpolated)

					grad = torch.autograd.grad(outputs=out,
											   inputs=interpolated,
											   grad_outputs=torch.ones(out.size()).cuda(),
											   retain_graph=True,
											   create_graph=True,
											   only_inputs=True)[0]

					grad = grad.view(grad.size(0), -1)
					grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
					d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

					# Backward + Optimize
					d_loss = self.lambda_gp * d_loss_gp

					self.reset_grad()
					d_loss.backward()
					self.d_optimizer.step()
			

			################################################
				self.dim_optimizer.zero_grad()
				sampler = random.randint(0,self.frames-1)  #16 frames
				pred_real = self.Dim(real_vid[:,:,sampler,:,:])
			
				if self.adv_loss == 'wgan-gp':
					pred_real = - torch.mean(pred_real)
				elif self.adv_loss == 'hinge':
					pred_real = torch.nn.ReLU()(1.0 - pred_real).mean()


				pred_fake = self.Dim(fake_images[:,:,sampler,:,:].detach())
				if self.adv_loss == 'wgan-gp':
					pred_fake = pred_fake.mean()
				elif self.adv_loss == 'hinge':
					pred_fake = torch.nn.ReLU()(1.0 + pred_fake).mean()


				# Backward + Optimize
				dim_loss = pred_real + pred_fake
				self.reset_grad()
				dim_loss.backward()
				self.dim_optimizer.step()

				#=======================================+++++#
				if self.adv_loss == 'wgan-gp':
					# Compute gradient penalty
					alpha = torch.rand(real_vid[:,:,sampler,:,:].size(0), 1, 1, 1).cuda().expand_as(real_vid[:,:,sampler,:,:])
					# print("sizes: ", real_vid.shape, fake_images.shape)
					interpolated = Variable(alpha * real_vid[:,:,sampler,:,:].data + (1 - alpha) * fake_images[:,:,sampler,:,:].data, requires_grad=True)
					out2 = self.Dim(interpolated)

					grad2 = torch.autograd.grad(outputs=out2,
											   inputs=interpolated,
											   grad_outputs=torch.ones(out2.size()).cuda(),
											   retain_graph=True,
											   create_graph=True,
											   only_inputs=True)[0]

					grad2 = grad2.view(grad2.size(0), -1)
					grad_l2norm2 = torch.sqrt(torch.sum(grad2 ** 2, dim=1))
					d_loss_gp2 = torch.mean((grad_l2norm2 - 1) ** 2)

					# Backward + Optimize
					d_loss2 = self.lambda_gp * d_loss_gp2

					self.reset_grad()
					d_loss2.backward()
					self.dim_optimizer.step()
				
			################################################

			# ================== Train G and gumbel ================== #
			# Create random noise
			z = tensor2var(torch.randn(real_vid.size(0), self.z_dim))
			sampler = random.randint(0,self.frames-1)  #16 frames
			fake_images = self.G(z)


			# Compute loss with fake images
			g_out_fake = self.D(fake_images)  # batch x n
			gim_out_fake = self.Dim(fake_images[:,:,sampler,:,:])

			if self.adv_loss == 'wgan-gp':
				g_loss_fake = - g_out_fake.mean()
				gim_loss = - gim_out_fake.mean()
			elif self.adv_loss == 'hinge':
				g_loss_fake = - g_out_fake.mean()
				gim_loss = - gim_out_fake.mean()
			g_loss = (g_loss_fake + gim_loss*self.gimlosswt)
			self.reset_grad()
			g_loss.backward()
		
			self.g_optimizer.step()

			# exit()
			# Print out log info
			if (step + 1) % self.log_step == 0:
				elapsed = time.time() - start_time
				elapsed = str(datetime.timedelta(seconds=elapsed))
				print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}".
					  format(elapsed, step + 1, self.total_step, (step + 1),
							 self.total_step , d_loss_real.data))

				writer.add_scalar('data/d_loss_real', d_loss_real.data, step)
				writer.add_scalar('data/d_loss_fake', d_loss_fake.data, step)
				writer.add_scalar('data/d_loss', d_loss.data, step)
				writer.add_scalar('data/d_lossgp', d_loss2.data, step)
				# writer.add_scalar('data/dim_loss', dim_loss.data, step)
				# writer.add_scalar('data/gim_loss', gim_loss.data, step)
				# writer.add_scalar('data/g_loss_fake', g_loss_fake.data, step)
				writer.add_scalar('data/g_loss', g_loss.data, step)
		
                

			# Sample images
			if (step + 1) % self.sample_step == 0:
				self.G.eval()
				fake_images = self.G(fixed_z)
				im = makeminimontage(fake_images.detach().cpu().numpy(), 2, 2, 2)
					
				tifdata = (im - im.min())/(im.max() - im.min())*255
				# print(tifdata.shape)
				with TiffWriter(os.path.join(self.sample_path, '{}_fake.tif'.format(step + 1)), bigtiff=True) as tif:
					for bk in range(tifdata.shape[1]):
						tif.save(tifdata[:,bk,:,:], compress=6)
				# plot_attn(ga1[0,:,:], ga2[0,:,:], self.log_path, step+1, 'ga1_2')
				# plot_attn(da1[0,:,:], da2[0,:,:], self.log_path, step+1, 'da1_2')
			# 	self.G.train()

			if (step+1) % model_save_step==0:
				print("Now saving models...")
				torch.save(self.G.state_dict(),
						   os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
				torch.save(self.D.state_dict(),
						   os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))
				torch.save(self.Dim.state_dict(),
						   os.path.join(self.model_save_path, '{}_Dim.pth'.format(step + 1)))

		writer.close()
		print("Now saving models for the last bit ...")
		torch.save(self.G.state_dict(),
			os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
		torch.save(self.D.state_dict(),
			os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))
		torch.save(self.D.state_dict(),
			os.path.join(self.model_save_path, '{}_Dim.pth'.format(step + 1)))
			

	def build_model(self):
		# torch.cuda.set_device(1)
		self.G = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).cuda()
		self.D = Discriminator(self.imsize, self.d_conv_dim).cuda()
		self.Dim = Discriminator2(1).cuda()
		if self.parallel:
			self.G = nn.DataParallel(self.G)
			self.D = nn.DataParallel(self.D)
			self.Dim = nn.DataParallel(self.Dim)

		# Loss and optimizer
		# self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
		self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
		self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])
		self.dim_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.Dim.parameters()), self.d_lr, [self.beta1, self.beta2])

		# self.c_loss = torch.nn.CrossEntropyLoss()
		# # print networks
		# print(self.G)
		# print(self.D)

	def build_tensorboard(self):
		from logger import Logger
		self.logger = Logger(self.log_path)

	def load_pretrained_model(self):
		self.G.load_state_dict(torch.load(os.path.join(
			self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
		self.D.load_state_dict(torch.load(os.path.join(
			self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
		self.Dim.load_state_dict(torch.load(os.path.join(
			self.model_save_path, '{}_Dim.pth'.format(self.pretrained_model))))
		print('loaded trained models (step: {})..!'.format(self.pretrained_model))

	def reset_grad(self):
		self.d_optimizer.zero_grad()
		self.g_optimizer.zero_grad()
		self.dim_optimizer.zero_grad()

	def save_sample(self, data_iter):
		real_vid, _ = next(data_iter)
		save_image(denorm(real_vid), os.path.join(self.sample_path, 'real.png'))
