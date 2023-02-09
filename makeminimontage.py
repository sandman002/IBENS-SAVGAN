import glob
import random
import os
import argparse
import sys
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np 
from PIL import Image
import torch
from skimage.external.tifffile import TiffFile
import tifffile
from skimage import io
from os import listdir
from os.path import isfile, join
import math
from skimage.external.tifffile import TiffWriter


def makeminimontage(fake, rowsep, colsep, numCol):

	numberofColumns=numCol
	totalfiles=fake.shape[0]
	numberofRows=math.ceil((totalfiles)/numberofColumns)

	rowseparation=rowsep #5pix
	columnseparation=colsep #2pix

	width = fake.shape[4]
	height= fake.shape[3]
	frames= fake.shape[2]
	channels = fake.shape[1]

	montageWidth = numberofColumns*(width) + (numberofColumns-1)*(columnseparation)
	montageHeight= numberofRows*height + (numberofRows-1)*rowseparation
	montage = np.zeros((channels, frames, montageHeight, montageWidth))
	bb = 0
	for ii in range(0,montageWidth, width+columnseparation):
		for jj in range(0,montageHeight, height+rowseparation):
			im =fake[bb,:,:,:,:]
			montage[:,:,jj:jj+height,ii:ii+width] = im
			bb = bb+1

	return montage