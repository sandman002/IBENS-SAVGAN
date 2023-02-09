import glob
import random
import os

from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np 
from PIL import Image
import torch
from skimage.external.tifffile import TiffFile
import tifffile
from skimage import io
from torchvideotransforms import video_transforms, volume_transforms
import torch.nn as nn
class AddGaussianNoise(object):
	def __init__(self, mean=0., std=1.):
		self.std = std
		self.mean = mean
		
	def __call__(self, tensor):
		return tensor + torch.randn(tensor.size()) * self.std + self.mean
	
	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class TemporalMean(object):
	def __init__(self, frame = 7):
		self.frame = frame
		
	def __call__(self, tensor):

		weights = torch.cuda.FloatTensor([[0., 0., 0.],
								[0., 1./float(self.frame), 0.],
								[0., 0., 0.]])
		weights = weights.view(1, 1, 1, 3, 3).repeat(1, 1, self.frame, 1, 1)

		t_mean = nn.Conv3d(1, 1, (self.frame,3,3), bias=False, padding=(int(self.frame/2),1,1))
		with torch.no_grad():
			t_mean.weight = nn.Parameter(weights)

		m = t_mean(tensor)

		return m
	
	def __repr__(self):
		return self.__class__.__name__ + '(frame={0})'.format(self.frame)

class ImageDataset(Dataset):
	def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
		self.transform = video_transforms.Compose(transforms_)
		self.unaligned = unaligned

		self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.tif'))
		# self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.tif'))

	def getVolume(self,filename):
		newupper = 1.
		newlower = -1.
		im = np.asarray(tifffile.imread(filename), dtype=np.float32)
		im = (im - im.min())*(newupper - newlower)/(im.max() - im.min()) + newlower
		
		return np.expand_dims(im, axis=0)


	def __getitem__(self, index):
		imA = self.getVolume(self.files_A[index % len(self.files_A)])
		_,nameA = os.path.split(self.files_A[index % len(self.files_A)])
		
		imA = np.swapaxes(imA, 0,1)
		imA = np.swapaxes(imA, 1,3)
		imA = np.swapaxes(imA, 1,2)

		item_A = self.transform(imA) #transform required	
	
		# imB = self.getVolume(self.files_B[index % len(self.files_B)])
		# _,nameB = os.path.split(self.files_B[index % len(self.files_B)])
		
		# imB = np.swapaxes(imB, 0,1)
		# imB = np.swapaxes(imB, 1,3)
		# imB = np.swapaxes(imB, 1,2)
		# item_B = self.transform(imB) #transform required

		
		# imB = self.getVolume(self.files_B[index % len(self.files_B)])

		# imB = np.swapaxes(imB, 0,1)
		# imB = np.swapaxes(imB, 1,3)
		# imB = np.swapaxes(imB, 1,2)

		# item_B = self.transform(imB) #transform required	
		return {'A': item_A, 'nameA': nameA[:-4]}

	def __len__(self):
		return (len(self.files_A))
