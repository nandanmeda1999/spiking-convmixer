import cupy
import pickle
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from spikingjelly.clock_driven import neuron, encoding, functional, layer, surrogate, base

parser = argparse.ArgumentParser(description='Spiking convmixer')
parser.add_argument('--dataset-dir', default = './', metavar='DIR', help='path to dataset')
parser.add_argument('--output-dir', default = './', metavar='DIR', help='path to directory in which output files shoule be saved')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=50, type=int, metavar='N')
parser.add_argument('--device', default='cuda:0', help='Device (supports only cuda)')

parser.add_argument('-T', '--timesteps', default=4, type=int, help='Simulation timesteps', dest='T')
parser.add_argument('--width', default=256, type=int, help='width / number of channels in the convolutional layers')
parser.add_argument('--depth', default=8, type=int, help='depth, the number of repetitions of the Spiking ConvMixer layer')
parser.add_argument('--kernel-size', default=9, type=int, help='kernel size of the depthwise convolutional layers')
parser.add_argument('--patch-size', default=1, type=int, help='patch size')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum factor')
parser.add_argument('--wd', '--weight-decay', default=0, type=float, metavar='WD', help='weight decay', dest='wd')
parser.add_argument('--epochs', default=100, type=int, help='training epochs')

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class SpikingConvMixer(nn.Module):
    def __init__(self, T, dim, depth, kernel_size, padding, patch_size, n_classes=10):
        super().__init__()
        self.T = T
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(dim)
        )
        conv = [neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy')]
        for _ in range(depth):
            conv.extend(self.conv_mixer_layer(dim, kernel_size, padding))
        conv.append(layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1,1))))
        self.conv = nn.Sequential(*conv)
        self.flatten = nn.Flatten(2)
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor):
        x = self.patch_embedding(x).unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x.mean(0))

    @staticmethod
    def conv_mixer_layer(dim, kernel_size, padding):
        return [ Residual(
            nn.Sequential(
              layer.SeqToANNContainer(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=padding),
                nn.BatchNorm2d(dim)
              ),
              neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy')
            )
          ),
          layer.SeqToANNContainer(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
          ),
          neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy')     
        ]

def main():
    _seed_ = 5
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    np.random.seed(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parser.parse_args()
    dataset_root_dir = args.dataset_dir
    output_dir = args.output_dir
    num_workers = args.workers
    batch_size = args.batch_size
    device = args.device
    T = args.T
    width = args.width
    depth = args.depth
    kernel_size = args.kernel_size
    # padding
    patch_size = args.patch_size
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.wd 
    epochs = args.epochs

    # implementing padding="same"
    if kernel_size % 2 == 0:
      padding = (int((kernel_size/2)-1), int(kernel_size/2))
    else:
      padding = int((kernel_size-1)/2)

    # Load data
    train_set = torchvision.datasets.CIFAR10(
      root=dataset_root_dir,
      train=True,
      transform=torchvision.transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
      ]),
      download=True)

    test_set = torchvision.datasets.CIFAR10(
      root=dataset_root_dir,
      train=False,
      transform=torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
      ]),
      download=True)
        
    train_data_loader = torch.utils.data.DataLoader(
      dataset=train_set,
      batch_size=batch_size, 
      shuffle=True, 
      pin_memory=True, 
      drop_last=True, 
      num_workers=num_workers)

    test_data_loader = torch.utils.data.DataLoader(
      dataset=test_set,
      batch_size=batch_size, 
      shuffle=False, 
      pin_memory=True, 
      drop_last=False, 
      num_workers=num_workers)

    net = SpikingConvMixer(T, width, depth, kernel_size, padding, patch_size)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_accuracies = []
    test_accuracies = []

    for epoch in range(0, epochs):
      net.train()
      train_accuracy = 0
      train_samples = 0
      for frame, label in train_data_loader:
        optimizer.zero_grad()
        frame = frame.float().to(device)
        label = label.to(device)
        out_fr = net(frame)
        loss = F.cross_entropy(out_fr, label)
        loss.backward()
        optimizer.step()
        train_samples += label.numel()
        train_accuracy += (out_fr.argmax(1) == label).float().sum().item()
        functional.reset_net(net)
      train_accuracy /= train_samples
      train_accuracies.append(train_accuracy)
      lr_scheduler.step()

      net.eval()
      test_accuracy = 0
      test_samples = 0
      with torch.no_grad():
        for frame, label in test_data_loader:
          frame = frame.float().to(device)
          label = label.to(device)
          out_fr = net(frame)
          loss = F.cross_entropy(out_fr, label)
          test_samples += label.numel()
          test_accuracy += (out_fr.argmax(1) == label).float().sum().item()
          functional.reset_net(net)
      test_accuracy /= test_samples
      test_accuracies.append(test_accuracy)

      print("Epoch: ", epoch, "Train accuracy:", train_accuracy, "Test accuracy:", test_accuracy) 

    # save train and test accuracies
    try:
      trainfile = open(os.path.join(output_dir, 'train_accuracies'), 'wb')
      pickle.dump(train_accuracies, trainfile)
      trainfile.close()
      testfile = open(os.path.join(output_dir, 'test_accuracies'), 'wb')
      pickle.dump(test_accuracies, testfile)
      testfile.close()
    except:
      print("Something went wrong")

if __name__ == '__main__':
    main()
