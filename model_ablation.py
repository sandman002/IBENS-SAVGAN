import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from spectral3 import SpectralNorm
import numpy as np
from torchsummary import summary
import torch.nn.utils.spectral_norm as spectral_norm
# from attention_nystrom2 import NystromAttention

landmarks = 64

# class deconv_block(nn.Module):

# 	def __init__(self, in_dim, out_dim, kernel_size=1, interpolation='bilinear', size=3, padding='same'):
# 		super(deconv_block, self).__init__()
# 		self.up = nn.Upsample(size = size, align_corners = True, mode=interpolation)
# 		self.conv3d = spectral_norm(nn.Conv3d(in_channels = in_dim, out_channels=out_dim, kernel_size = kernel_size))
# 		self.batchnorm = nn.InstanceNorm3d(out_dim)
# 		self.relu = nn.ReLU()

# 	def forward(self,x):
# 		y = self.up(x)
# 		y = self.conv3d(y)
# 		y = self.batchnorm(y)
# 		y = self.relu(y)

# 		return y

class Self_Attn(nn.Module):
	def __init__(self, in_dim, activation):
		super(Self_Attn, self).__init__()
		self.channel_in = in_dim
		self.activation = activation

		self.query_conv = nn.Conv3d(in_channels = in_dim, out_channels=in_dim//8, kernel_size = 1)
		self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels = in_dim//8, kernel_size = 1)
		self.value_conv = nn.Conv3d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)
		self.gamma = nn.Parameter(torch.zeros(1))

		self.softmax = nn.Softmax(dim=-1)

	def forward(self,x):
		'''
			inputs:
				x: BxCxTxHxW
			returns:
				out: self attention value+input_features
				attention: BxNxNXN 
		'''
		m_batchsize,C, T, width ,height = x.size()
		# print(x.shape)
		proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height*T)
		
		proj_query = proj_query.permute(0,2,1) # B X CX(N)


		proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height*T) # B X C x (*W*H)
		energy =  torch.bmm(proj_query,proj_key) # transpose check
		attention = self.softmax(energy) # BX (N) X (N) 
		proj_value = self.value_conv(x).view(m_batchsize,-1,width*height*T) # B X C X N
		
		out = torch.bmm(proj_value,attention.permute(0,2,1))

		out = out.view(m_batchsize,C,T,width,height)

		out = self.gamma*out + x
		return out,attention

class Self_nysAttn(nn.Module):
	def __init__(self, in_dim, num_landmarks, seq_len):
		super(Self_nysAttn, self).__init__()
		self.channel_in = in_dim

		self.query_conv = nn.Conv3d(in_channels = in_dim, out_channels=in_dim//8, kernel_size = 1)
		self.key_conv  = nn.Conv3d(in_channels = in_dim, out_channels=in_dim//8, kernel_size = 1)
		self.value_conv = nn.Conv3d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)
		self.gamma = nn.Parameter(torch.zeros(1))

		self.softmax = nn.Softmax(dim=-1)
		self.NysAttn = NystromAttention(head_dim = in_dim//8, \
			num_landmarks = num_landmarks, \
			seq_len = seq_len)  



	def forward(self,x):
		
			# inputs:
			# 	x: BxCxTxHxW
			# returns:
			# 	out: self attention value+input_features
			# 	attention: BxNxNXN 
		
		m_batchsize,C, T, width ,height = x.size()
		
		proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height*T)
		proj_query = proj_query.permute(0,2,1) # B X CX(N)
		proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height*T)
		proj_value = self.value_conv(x).view(m_batchsize,-1,width*height*T) # B X C X N


		# print("Input shape:", proj_query.shape, proj_key.shape, proj_value.shape )
		attn = self.NysAttn(proj_query, proj_key.permute(0,2,1))
		out = torch.bmm(proj_value,attn.permute(0,2,1))
		out = out.view(m_batchsize,C,T, width,height)
		out = self.gamma*out + x
		return out,attn#,self.gamma, fattn



class Generator(nn.Module):
	"""Generator."""

	def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64):
		super(Generator, self).__init__()
		self.imsize = image_size
		layer1 = []
		layer2 = []
		layer3 = []
		last = []

		repeat_num = int(np.log2(self.imsize)) - 3
		mult = 2 ** repeat_num # 8
		layer1.append((nn.ConvTranspose3d(z_dim, conv_dim * mult, (2,4,4))))
		layer1.append(nn.InstanceNorm3d(conv_dim * mult))
		layer1.append(nn.ReLU())

		curr_dim = conv_dim * mult

		layer2.append(spectral_norm(nn.ConvTranspose3d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
		layer2.append(nn.InstanceNorm3d(int(curr_dim / 2)))
		layer2.append(nn.ReLU())

		curr_dim = int(curr_dim / 2)

		layer3.append(spectral_norm(nn.ConvTranspose3d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
		layer3.append(nn.InstanceNorm3d(int(curr_dim / 2)))
		layer3.append(nn.ReLU())

		if self.imsize >= 64:
			layer4 = []
			curr_dim = int(curr_dim / 2)
			layer4.append(spectral_norm(nn.ConvTranspose3d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
			layer4.append(nn.InstanceNorm3d(int(curr_dim / 2)))
			layer4.append(nn.ReLU())

			self.l4 = nn.Sequential(*layer4)
			curr_dim = int(curr_dim / 2)

		self.l1 = nn.Sequential(*layer1)
		self.l2 = nn.Sequential(*layer2)
		self.l3 = nn.Sequential(*layer3)

		last.append(nn.ConvTranspose3d(curr_dim, 1, 4, 2, 1))
		last.append(nn.Tanh())
		self.last = nn.Sequential(*last)

		# self.attn1 = Self_Attn( 64, 'relu')
		# self.attn2 = Self_Attn( 32,  'relu')

		# self.attn1 = Self_nysAttn( 64, landmarks,  16*16*8)
		# self.attn2 = Self_nysAttn( 32,  landmarks, 32*32*16)

	def forward(self, z):
		z = z.view(z.size(0), z.size(1), 1, 1, 1)
		out=self.l1(z)
		out=self.l2(out)
		
		out=self.l3(out)
		# print("GEN1: ", out.shape)
		# out,p1 = self.attn1(out)
		
		
		out=self.l4(out)
		# print("GENa1: ", out.shape)
		# out,p2 = self.attn2(out)

		out=self.last(out)

		return out#, p1, p2

class Discriminator(nn.Module):

	def __init__(self, image_size=64, conv_dim=64):
		super(Discriminator, self).__init__()
		self.imsize = image_size
		layer1 = []
		layer2 = []
		layer3 = []
		last = []

		layer1.append(spectral_norm(nn.Conv3d(1, conv_dim, (4,4,4), (1,2,2), 1)))
		layer1.append(nn.InstanceNorm3d(conv_dim))
		layer1.append(nn.LeakyReLU(0.1))

		curr_dim = conv_dim

		layer2.append(spectral_norm(nn.Conv3d(curr_dim, curr_dim*2, 4, 2, 1)))
		layer2.append(nn.InstanceNorm3d(curr_dim*2))
		layer2.append(nn.LeakyReLU(0.1))
		curr_dim = curr_dim*2

		layer3.append(spectral_norm(nn.Conv3d(curr_dim, curr_dim*2, 4, 2, 1)))
		layer3.append(nn.InstanceNorm3d(curr_dim*2))
		layer3.append(nn.LeakyReLU(0.1))
		curr_dim = curr_dim*2

		if self.imsize >= 64:
			layer4= []
			layer4.append(spectral_norm(nn.Conv3d(curr_dim, curr_dim*2, 4, 2, 1)))
			layer4.append(nn.InstanceNorm3d(curr_dim*2))
			layer4.append(nn.LeakyReLU(0.1))
			self.l4 = nn.Sequential(*layer4)
			curr_dim = curr_dim*2
		self.l1 = nn.Sequential(*layer1)
		self.l2 = nn.Sequential(*layer2)
		self.l3 = nn.Sequential(*layer3)

		last.append(nn.Conv3d(curr_dim, 1, 3))
		self.last = nn.Sequential(*last)

		# self.attn1 = Self_nysAttn(128, landmarks, 7*8*8)
		# self.attn2 = Self_nysAttn(256, 16, 3*4*4)

		# self.attn1 = Self_Attn( 128, 'relu')
		# self.attn2 = Self_Attn( 256,  'relu')

	def forward(self, x):
		out = self.l1(x)
		out = self.l2(out)
		out = self.l3(out)
		# print("DIS1", out.shape)
		# out, p1= self.attn1(out)
		
		out = self.l4(out)
		# print("DIS2", out.shape)
		# out, p2= self.attn2(out)
		out = self.last(out)
		# print("DisaOUT: ", out.shape)
		return out.squeeze(), p1, p2# torch.cuda.set_device(2)
# Tensor = torch.cuda.FloatTensor
# fixed_noise = Tensor(np.random.normal(0,1,(1,1, 32,128,128)))
# netD = Discriminator(image_size=64, conv_dim=32)
# netD.cuda()
# out, p1, p2 = netD(fixed_noise)

# print(out, p1.shape, p2.shape)
# summary(netD, (1,32,128,128))