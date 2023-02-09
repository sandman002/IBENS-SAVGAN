
import os
import time
import torch
import datetime

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

# from myDis import Generator
# from myDis import Discriminator
from model_full import Generator
# from model_actual import Generator
# from model_ablation import Generator
# from model_ablation import Discriminator
import numpy as np
from utils import *
from skimage.external import tifffile as tif
from makeminimontage import makeminimontage
from skimage.external.tifffile import TiffWriter
Tensor = torch.cuda.FloatTensor
class Tester(object):
	def __init__(self, config):


		# exact model and loss
		self.model = config.model
	
		# Model hyper-parameters
		self.imsize = config.imsize
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
		self.test_model = config.test_model

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

		

		# Path
		self.log_path = os.path.join(config.log_path, self.version)
		self.sample_path = os.path.join(config.sample_path, 'test')
		self.model_save_path = os.path.join(config.model_save_path, 'test')

		self.build_model()
		self.load_test_model()

	

	def build_model(self):
		torch.cuda.set_device(0)
		self.G = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).cuda()
		# self.D = Discriminator(self.imsize, self.d_conv_dim).cuda()
		# if self.parallel:
		# 	print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$In parallel mode")
		self.G = nn.DataParallel(self.G)
			# self.D = nn.DataParallel(self.D)
# /home/smanandh/totem/sandeep/Self-Attention-GAN-master/models_2/sagan_1/89300_G.pth

	def load_test_model(self):
		#/home/smanandh/totem/sandeep/Self-Attention-GAN-master/models/sagan_1/364374_G.pth
		# self.G.load_state_dict(torch.load(os.path.join(
		# 	self.model_save_path, '{}_G.pth'.format(self.test_model))))

		# self.G.load_state_dict(torch.load('./models_act/74970_G.pth'))
		self.G.load_state_dict(torch.load('./models_full/sagan_1/234090_G.pth'))
		# print('loaded trained models (step: {})..!'.format(self.test_model))

		# state_dict=(torch.load('/projects/totem/sandeep/Self-Attention-GAN-master/models/sagan_1/364374_G.pth'))		
		# from collections import OrderedDict
		# new_state_dict = OrderedDict()
		# state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

		# self.G.load_state_dict(state_dict)

	def save_sample(self, data_iter):
		real_images, _ = next(data_iter)
		save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))

	def test(self,batch_num):
		bs = 8
		z = tensor2var(torch.randn(bs, self.z_dim))
		self.G.eval()
		with torch.no_grad():
			fake_images = self.G(z)
		fake_images = fake_images.cpu().numpy()
		# im = makeminimontage(fake_images.detach().cpu().numpy(), 2, 2, 8)
		# tifdata = (im - im.min())/(im.max() - im.min())*255
		# tifdata=tifdata.astype(np.uint8)
		# with TiffWriter(os.path.join(self.sample_path, '{}_fake.tif'.format(batch_num + 1)), bigtiff=True) as tif:
		# 	for bk in range(tifdata.shape[0]):
		# 		tif.save(tifdata[bk], compress=6)
		for i in range(bs):
			im = fake_images[i,:,:,:,:]
			print(im.shape)

			tifdata = (im - im.min())/(im.max() - im.min())*255
			tifdata=tifdata.astype(np.uint8)
			tif.imsave(os.path.join(self.sample_path, '{}_{}'.format(batch_num,  i)) + '.tif', np.asarray(im), bigtiff=True)