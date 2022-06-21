import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from math import log10
from model import ZebraSRNet
from dataset import datasetTrain, datasetVal
import argparse
import os
import numpy
import random

#===== Training settings =====#
parser = argparse.ArgumentParser(description='NTHU EE - CP HW3 - ZebraSRNet')
parser.add_argument('--patchSize', type=int, default=64, help='HR image cropping (patch) size for training')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--epochSize', type=int, default=150, help='number of batches as one epoch (for validating once)')
parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs for training')
parser.add_argument('--nFeat', type=int, default=16, help='channel number of feature maps')
parser.add_argument('--ExpandRatio', type=int, default=3, help='expansion ratio of residual block')
parser.add_argument('--nResBlock', type=int, default=2, help='number of residual blocks')
parser.add_argument('--nTrain', type=int, default=2, help='number of training images')
parser.add_argument('--nVal', type=int, default=1, help='number of validation images')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use, if Your OS is window, please set to 0')
parser.add_argument('--seed', type=int, default=777, help='random seed to use. Default=777')
parser.add_argument('--printEvery', type=int, default=30, help='number of batches to print average loss ')
args = parser.parse_args()

print(args)

print(torch.__version__)

if args.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed)
torch.backends.cudnn.benchmark = False

#===== Datasets =====#
def seed_worker(worker_id):
    worker_seed = args.seed
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    
print('===> Loading datasets')
train_set = datasetTrain(args)
train_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True, worker_init_fn=seed_worker)
val_set = datasetVal(args)
val_data_loader = DataLoader(dataset=val_set, num_workers=args.threads, batch_size=1, shuffle=False, worker_init_fn=seed_worker)

#===== ZebraSRNet model =====#
print('===> Building model')
net = ZebraSRNet(nFeat=args.nFeat, ExpandRatio=args.ExpandRatio, nResBlock=args.nResBlock)

if args.cuda:
    net = net.cuda()

#===== Loss function and optimizer =====#
criterion = torch.nn.L1Loss()

if args.cuda:
    criterion = criterion.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

#===== Training and validation procedures =====#
def train(f, epoch):
    net.train()
    epoch_loss = 0
    for iteration, batch in enumerate(train_data_loader):
        varIn, varTar = Variable(batch[0]), Variable(batch[1])
        if args.cuda:
            varIn = varIn.cuda()
            varTar = varTar.cuda()

        optimizer.zero_grad()
        loss = criterion(net(varIn), varTar)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
        if (iteration+1)%args.printEvery == 0:
            print("===> Epoch[{}]({}/{}): Avg. Loss: {:.4f}".format(epoch, iteration+1, len(train_data_loader), epoch_loss/args.printEvery))
            f.write("===> Epoch[{}]({}/{}): Avg. Loss: {:.4f}\n".format(epoch, iteration+1, len(train_data_loader), epoch_loss/args.printEvery))
            epoch_loss = 0

def validate(f):
    net.eval()
    avg_psnr = 0
    mse_criterion = torch.nn.MSELoss()
    for batch in val_data_loader:
        varIn, varTar = Variable(batch[0]), Variable(batch[1])
        if args.cuda:
            varIn = varIn.cuda()
            varTar = varTar.cuda()

        prediction = net(varIn)
        prediction[prediction>  1] =   1
        prediction[prediction<  0] =   0
        mse = mse_criterion(prediction, varTar)
        psnr = 10 * log10(1.0*1.0/mse.data)
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(val_data_loader)))
    f.write("===> Avg. PSNR: {:.4f} dB\n".format(avg_psnr / len(val_data_loader)))

#===== Model saving =====#
save_dir = './model_trained'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

def checkpoint(epoch): 
    save_name = 'net_F{}B{}E{}_epoch_{}.pth'.format(args.nFeat, args.nResBlock, args.ExpandRatio, epoch)
    save_path = os.path.join(save_dir, save_name)
    torch.save(net, save_path)
    print("Checkpoint saved to {}".format(save_path))

#===== Main procedure =====#
with open('train_net_F{}B{}E{}.log'.format(args.nFeat, args.nResBlock, args.ExpandRatio),'w') as f:
    f.write('training log record of F={}, B={}, E={}, random seed={}\n'.format(args.nFeat, args.nResBlock, args.ExpandRatio, args.seed))
    f.write('dataset configuration: epoch size = {}, batch size = {}, patch size = {}\n'.format(args.epochSize, args.batchSize, args.patchSize))
    print('-------')
    for epoch in range(1, args.nEpochs+1):
        train(f, epoch)
        validate(f)
        checkpoint(epoch)
