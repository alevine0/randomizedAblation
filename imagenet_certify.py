from __future__ import print_function

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets

import torchvision
import torchvision.transforms as transforms
import resnet_imgnt as resnet

import os
import argparse

from prog_bar import progress_bar
import utils_color as utils

parser = argparse.ArgumentParser(description='L0 Certificate Evaluation')
parser.add_argument('--keep',  required=True, type=int, help='pixels to keep')
parser.add_argument('--model',  required=True, help='checkpoint to certify')
parser.add_argument('--alpha', default=0.05, type=float, help='Certify to 1-alpha probability')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--predsamples', default=1000, type=int, help='samples for prediction')
parser.add_argument('--boundsamples', default=10000, type=int, help='samples for bound')
parser.add_argument('--valpath', default='imagenet-val/val', type=str, help='Path to ImageNet validation set')

args = parser.parse_args()
checkpoint_dir = 'checkpoints'
radii_dir = 'radii'
if not os.path.exists('./radii'):
    os.makedirs('./radii')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_indices = torch.load('imagenet_indices.pth')

# Data
print('==> Building model..')


valdir = args.valpath
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


testloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=1, shuffle=False,
    num_workers=2, pin_memory=True,sampler=torch.utils.data.sampler.SubsetRandomSampler(test_indices))

# Model
print('==> Building model..')

net = resnet.resnet50()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
resume_file = '{}/{}'.format(checkpoint_dir, args.model)
assert os.path.isfile(resume_file)
checkpoint = torch.load(resume_file)
net.load_state_dict(checkpoint['net'])

net.eval()
all_batches = []
for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.no_grad():
        #breakpoint()
        batch_radii = utils.certify(inputs, targets, net, args.alpha, args.keep, args.predsamples, args.boundsamples,sub_batch=1000)
        all_batches.append(batch_radii)
        progress_bar(batch_idx, len(testloader))
out = torch.cat(all_batches)
sortd,indices = torch.sort(out)
torch.save(sortd, radii_dir +'/'+args.model+'_alpha_'+str(args.alpha)+'_boundsamples_'+str(args.boundsamples)+'_predsamples_'+str(args.predsamples) + '.pth')
