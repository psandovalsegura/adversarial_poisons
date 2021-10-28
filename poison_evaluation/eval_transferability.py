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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    parser.add_argument('poison_path', type=str, help='path to the poison dataset')
    parser.add_argument('--cifar_path', type=str, default='/vulcanscratch/psando/cifar-10')
    parser.add_argument('--ckpt_dir', type=str, default='/vulcanscratch/psando/cifar_model_ckpts/')
    parser.add_argument('--adv_ckpt_dir', type=str, default='/vulcanscratch/psando/cifar_model_ckpts/adv')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--disable_tqdm', action='store_true')
    args = parser.parse_args()
    print(args)
    evaluate(args)

def test_transferability(net, net_name, poison_loader, disable_tqdm):
    """ Compute the proportion of clean samples from the training set
        which the model classifies correctly, but where the corresponding
        poison image causes a misclassification.
    """
    net.eval()
    success = 0
    initially_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(poison_loader), desc='Poison Transfer', disable=disable_tqdm):
            inputs, targets, clean_inputs = batch
            inputs, targets, clean_inputs = inputs.to(device), targets.to(device), clean_inputs.to(device)
            adv_outputs = net(inputs)
            adv_pred = torch.argmax(adv_outputs, dim=1)
            clean_outputs = net(clean_inputs)
            clean_pred = torch.argmax(clean_outputs, dim=1)

            transfer_success = torch.eq(torch.ne(adv_pred, targets), torch.eq(clean_pred, targets))
            num_success = transfer_success.sum().item()
            num_initially_correct = torch.eq(clean_pred, targets).sum().item()
            success += num_success
            initially_correct += num_initially_correct
            total += inputs.size(0)
    print('*'*50)
    print(f'Model: {net_name}')
    print(f'Transfer Success Rate: {success * 1.0 / initially_correct * 100 :.2f} % ({success} / {initially_correct})')
    print(f'Clean Accuracy: {initially_correct * 1.0 / total * 100 :.2f} % ({initially_correct} / {total})')
    print('*'*50)

def evaluate(args):
    # Data
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    
    baseset = torchvision.datasets.CIFAR10(
    root=args.cifar_path, train=True, download=False, transform=transform_test)
    trainset = CIFAR_load(root=args.poison_path, baseset=baseset)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=args.workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    accs = []
    model_names = ['resnet18', 'vgg19', 'mobilenet', 'googlenet']
    use_adv_trained = [False, True]
    for use_adv in use_adv_trained:
        for model_name in model_names:
            # Model
            print('==> Building model..')
            if str.lower(model_name) == 'vgg19':
                net = VGG('VGG19')
            elif str.lower(model_name) == 'resnet18':
                net = ResNet18()
            elif str.lower(model_name) == 'googlenet':
                net = GoogLeNet()
            elif str.lower(model_name) == 'mobilenet':
                net = MobileNet()
            elif str.lower(model_name) == 'efficientnetb0':
                net = EfficientNetB0()
            elif str.lower(model_name) == 'densenet121':
                net = DenseNet121()
            else:
                raise NotImplementedError(f'Model {model_name} not implemented.')

            net = net.to(device)
            if device == 'cuda':
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = True

            # Load pretrained model
            if use_adv:
                ckpt_path = os.path.join(args.adv_ckpt_dir, f'{model_name}.pt')
                ckpt = torch.load(ckpt_path)
                net_state_dict = ckpt['model']
                net.module.load_state_dict(net_state_dict)
            else:
                ckpt_path = os.path.join(args.ckpt_dir, f'{model_name}.pt')
                ckpt = torch.load(ckpt_path)
                net_state_dict = ckpt['model']
                net.load_state_dict(net_state_dict)
            epoch = ckpt['epoch']
            print(f'==> Loaded checkpoint from {ckpt_path} (epoch {epoch}).')

            print('==> Evaluating poison transferability')
            model_name = model_name if not use_adv else f'adv-{model_name}'
            test_transferability(net, model_name, trainloader, disable_tqdm=args.disable_tqdm)



if __name__ == '__main__':
    main()