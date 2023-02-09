import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np

class ImagePool():

    def __init__(self, pool_size):

        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_images = 0
            self.images = []

    def query(self, images):
        ''''' Return images from the pool.
        Input: the latest generated images from the generator

        return 50% latest images and 50% images from previous iteration

        '''
        if self.pool_size == 0:
            return images
        return_images=[]
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_images < self.pool_size:
                self.num_images = self.num_images + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0,1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id]=image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images,0)
        return return_images




class ReflectionPad3d(torch.nn.modules.padding._ReflectionPadNd):
    def __init__(self, padding):
        super(ReflectionPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)




def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm3d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)




def MIPTensor(im, im_size):

    im = im.view(im_size[0],im_size[1],im_size[2])
    maximage = (im.cpu().data.numpy())
    maximage = (maximage - maximage.min())/(maximage.max() - maximage.min())*255
    # maximage = (np.squeeze(maximage, axis=0))
    mip = np.amax(maximage, axis=0)

    return mip.astype(np.uint8)



class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)




class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1 # set accordingly
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}



    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
           
            im_size = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
            if image_name not in self.image_windows:
                
                #If your tensor is simply an image, you do not need to call MIPTensor routine
                self.image_windows[image_name] = self.viz.image(MIPTensor(tensor[0,:,:,:,:], im_size), opts={'title':image_name}) 
            else:
                self.viz.image(MIPTensor(tensor[0,:,:,:,:], im_size), win=self.image_windows[image_name], opts={'title':image_name})

        
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name}) #Labels of axes to display
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

        # End of epoch
            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1