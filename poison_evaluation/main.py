'''Train CIFAR10 with PyTorch.'''
from re import T
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np

from models import *
#from utils import progress_bar
from PIL import Image
from tqdm import tqdm

class CIFAR_load(torch.utils.data.Dataset):
    def __init__(self, root, baseset):

        self.baseset = baseset
        self.transform = self.baseset.transform
        self.samples = os.listdir(os.path.join(root, 'data'))
        self.root = root

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        true_index = int(self.samples[idx].split('.')[0])
        true_img, label = self.baseset[true_index]
        return self.transform(Image.open(os.path.join(self.root, 'data',
                                            self.samples[idx]))), label, true_img

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--poison_path', type=str, default=None)
    parser.add_argument('--cifar_path', type=str)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--use_wd', action='store_true', help='whether to use weight decay')
    parser.add_argument('--use_scheduler', action='store_true', help='whether to use learning rate scheduler')
    args = parser.parse_args()
    print(args)
    train(args)

def train(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0     # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    
    
    if args.poison_path is not None:
        baseset = torchvision.datasets.CIFAR10(
        root=args.cifar_path, train=True, download=False, transform=transform_test)
        trainset = CIFAR_load(root=args.poison_path, baseset=baseset)
    else: 
        baseset = torchvision.datasets.CIFAR10(
        root=args.cifar_path, train=True, download=False, transform=transform_train)
        trainset = torchvision.datasets.CIFAR10(
        root=args.cifar_path, train=True, download=False, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(
        root=args.cifar_path, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=args.workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    accs = []

    for run in range(args.runs):
        # Model
        print('==> Building model..')
        if str.lower(args.model_name) == 'vgg19':
            net = VGG('VGG19')
        elif str.lower(args.model_name) == 'resnet18':
            net = ResNet18()
        elif str.lower(args.model_name) == 'resnet50':
            net = ResNet50()
        elif str.lower(args.model_name) == 'googlenet':
            net = GoogLeNet()
        elif str.lower(args.model_name) == 'mobilenet':
            net = MobileNet()
        elif str.lower(args.model_name) == 'efficientnetb0':
            net = EfficientNetB0()
        elif str.lower(args.model_name) == 'densenet121':
            net = DenseNet121()
        
        # net = PreActResNet18()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()
        # net = ShuffleNetV2(1)
        # net = EfficientNetB0()
        # net = RegNetX_200MF()
        #net = SimpleDLA()

        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        # Load pretrained model
        if args.use_pretrained:
            ckpt_path = os.path.join(args.ckpt_dir, f'{args.model_name}.pt')
            ckpt = torch.load(ckpt_path)
            net_state_dict = ckpt['model']
            net.load_state_dict(net_state_dict)
            print(f'==> Loaded checkpoint from {ckpt_path}.')


        # Training
        def train(epoch):
            print('\nEpoch: %d' % epoch)
            net.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, batch in tqdm(enumerate(trainloader), desc=f'Train Epoch {epoch}', disable=args.disable_tqdm):
                if args.poison_path:
                    inputs, targets, clean_inputs = batch
                    inputs, targets, clean_inputs = inputs.to(device), targets.to(device), clean_inputs.to(device)
                else:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        def test(epoch, loader, name):
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, batch in tqdm(enumerate(loader), desc=f'Test Epoch {epoch}', disable=args.disable_tqdm):
                    if name == 'Train' and args.poison_path:
                        inputs, targets, clean_inputs = batch
                    else:
                        inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                print(f'{name} avg loss: {test_loss/(batch_idx + 1)}, {name} acc: {100. * correct/total}')

            # Save checkpoint.
            acc = 100.*correct/total
            return acc

        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=(5e-4 if args.use_wd else 0))
        if args.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        else:
            scheduler = None
        for epoch in range(start_epoch, start_epoch+args.epochs):
            train(epoch)
            acc = test(epoch, testloader, 'Test')
            _ = test(epoch, trainloader, 'Train')
            if scheduler:
                scheduler.step()

            if epoch % 10 == 0 and (args.ckpt_dir is not None):
                print(f'==> Saving best checkpoint to: {args.ckpt_dir}')
                sd_info = {
                    'model':net.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'schedule':(scheduler.state_dict() if scheduler else None),
                    'epoch':epoch+1,
                }

                ckpt_save_path = os.path.join(args.ckpt_dir, f'{args.model_name}.pt')
                torch.save(sd_info, ckpt_save_path)

            if epoch == start_epoch + args.epochs - 1:
                accs.append(acc)

    print(f'Poison: {args.poison_path}')
    print(f'Mean accuracy: {np.mean(np.array(accs))}, \
                Std_error: {np.std(np.array(accs))/np.sqrt(args.runs)}')

if __name__ == '__main__':
    main()