
from parameter import *
from trainer_full import Trainer
# from tester import Tester

from torch.backends import cudnn
from utils import make_folder
from datasets_BC import ImageDataset
from torchvideotransforms import video_transforms, volume_transforms
from torch.utils.data import DataLoader
import os



def main(config):
    # For fast training
    cudnn.benchmark = True


    # Data loader
    # data_loader = Data_Loader(config.train, config.dataset, config.image_path, config.imsize,
    #                          config.batch_size, shuf=config.train)

    video_transform_list = [\
        volume_transforms.ClipToTensor(channel_nb=1, div_255=False, numpy=True),\
        ]
    dataroot = config.image_path

    n_cpu = 8
    data_loader = DataLoader(ImageDataset(dataroot, mode = 'train',transforms_=video_transform_list, unaligned=True),
    batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=n_cpu)


    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)


    if config.train:
        if config.model=='sagan':
            trainer = Trainer(data_loader, config)
        elif config.model == 'qgan':
            trainer = qgan_trainer(data_loader, config)
        trainer.train()
    # else:
    #     tester = Tester(data_loader.loader(), config)
    #     tester.test()

if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)

    #batch4 30533MiB