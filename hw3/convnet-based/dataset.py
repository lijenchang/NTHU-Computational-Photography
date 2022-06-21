import torch
import torch.utils.data as data
import random
import imageio
import os
import numpy as np

random.seed(777)
np.random.seed(777)
torch.manual_seed(777)

# ===== Utility functions for data augment =====#
def randomCrop(imgIn, imgTar, patchSize, scale=4):
    (ih, iw, c) = imgIn.shape
    (th, tw) = (scale * ih, scale * iw)

    tp = patchSize
    ip = tp // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    (tx, ty) = (scale * ix, scale * iy)
    imgIn = imgIn[iy:iy + ip, ix:ix + ip, :]
    imgTar = imgTar[ty:ty + tp, tx:tx + tp, :]

    return imgIn, imgTar

def np2PytorchTensor(imgIn, imgTar):
    ts = (2, 0, 1)
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float))
    imgTar = torch.Tensor(imgTar.transpose(ts).astype(float))

    return imgIn, imgTar

def augment(imgIn, imgTar, hflip=True, vflip=True, rotation=True):
    if random.random() < 0.5 and hflip:
        imgIn = imgIn[:, ::-1, :]
        imgTar = imgTar[:, ::-1, :]

    if random.random() < 0.5 and vflip:
        imgIn = imgIn[::-1, :, :]
        imgTar = imgTar[::-1, :, :]

    if random.random() < 0.5 and rotation:
        imgIn = imgIn.transpose(1, 0, 2)
        imgTar = imgTar.transpose(1, 0, 2)

    return imgIn, imgTar

# ===== Dataset =====#
class datasetTrain(data.Dataset):
    def __init__(self, args):
        self.patchSize = args.patchSize
        self.epochSize = args.epochSize
        self.batchSize = args.batchSize
        self.nTrain = args.nTrain
        self.trainDir = 'image_train'
        self.imgInPrefix = 'LR_zebra_train'
        self.imgTarPrefix = 'HR_zebra_train'   

    def __getitem__(self, idx):
        idx = (idx % self.nTrain) + 1

        nameIn, nameTar = self.getFileName(idx)
        imgIn = imageio.imread(nameIn)/255.0
        imgTar = imageio.imread(nameTar)/255.0
 
        imgIn, imgTar = randomCrop(imgIn, imgTar, self.patchSize)
        imgIn, imgTar = augment(imgIn, imgTar)

        return np2PytorchTensor(imgIn, imgTar)

    def __len__(self):
        return self.epochSize*self.batchSize

    def getFileName(self, idx):
        fileName = '{:0>4}'.format(idx)
        nameIn = '{}_{}.png'.format(self.imgInPrefix, fileName)
        nameIn = os.path.join(self.trainDir, nameIn)
        nameTar = '{}_{}.png'.format(self.imgTarPrefix, fileName)
        nameTar = os.path.join(self.trainDir, nameTar)

        return nameIn, nameTar

class datasetVal(data.Dataset):
    def __init__(self, args):
        self.nVal = args.nVal
        self.valDir = 'image_val'
        self.imgInPrefix = 'LR_zebra_val'
        self.imgTarPrefix = 'HR_zebra_val'

    def __getitem__(self, idx):
        idx = (idx % self.nVal) + 1

        nameIn, nameTar = self.getFileName(idx)
        imgIn = imageio.imread(nameIn)/255.0
        imgTar = imageio.imread(nameTar)/255.0

        return np2PytorchTensor(imgIn, imgTar)

    def __len__(self):
        return self.nVal

    def getFileName(self, idx):
        fileName = '{:0>4}'.format(idx)
        nameIn = '{}_{}.png'.format(self.imgInPrefix, fileName)
        nameIn = os.path.join(self.valDir, nameIn)
        nameTar = '{}_{}.png'.format(self.imgTarPrefix, fileName)
        nameTar = os.path.join(self.valDir, nameTar)

        return nameIn, nameTar