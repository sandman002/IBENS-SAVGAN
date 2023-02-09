
from parameter import *

from tester import Tester
# from data_loader import Data_Loader
from torch.backends import cudnn
from utils import make_folder
from datasets_BC import ImageDataset
from torchvideotransforms import video_transforms, volume_transforms
from torch.utils.data import DataLoader
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

def main(config):
    # For fast training
    cudnn.benchmark = True


    tester = Tester(config)
    for i in range(32):
    	tester.test(i)

if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)