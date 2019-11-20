from __future__ import print_function

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
from prog_bar import progress_bar
import utils
import foolbox_utils
import foolbox

parser = argparse.ArgumentParser(description='L0 Robustness Evaluation')
parser.add_argument('--keep',  required=True, type=int, help='pixels to keep')
parser.add_argument('--model',  required=True, help='checkpoint to certify')
parser.add_argument('--alpha', default=0.05, type=float, help='Certify to 1-alpha probability')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--start', default=0, type=int, help='start index')
parser.add_argument('--repetitions', default=10, type=int, help='number of times to repeat attack')
parser.add_argument('--samples', default=10000, type=int, help='samples')

args = parser.parse_args()
checkpoint_dir = 'checkpoints'
empirical_dir = 'empirical'
if not os.path.exists('./empirical'):
    os.makedirs('./empirical')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

# Model
print('==> Building model..')

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

net = nn.Sequential(
        nn.Conv2d(2, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(128*7*7,500),
        nn.ReLU(),
        nn.Linear(500,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

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
model = foolbox_utils.AblatedTorchModel(net,args.samples,args.keep,(0, 1),10)
haattack = foolbox.attacks.PointwiseAttack(model=model, criterion=foolbox_utils.MisclassificationOrAbstain(args.samples, args.alpha), distance=foolbox.distances.L0)
for batch_idx in range(args.start, 10000):
    (inputs, targets) = testset[batch_idx]
    inpt, target = inputs.numpy(), targets
    print(target)
    with torch.no_grad():
        adversarial_dists = []
        adversarials = []
        for x in range(args.repetitions):
            adv = haattack(inpt, label=target, unpack=False)
            adversarial_dists.append(adv.distance.value)
            adversarials.append({'image': adv.image, 'original_image': adv.original_image,'distance': adv.distance.value,'label': adv.original_class, 'scores':  adv.output,'index':batch_idx})
        mindex = np.argmin(adversarial_dists)
        torch.save(adversarials[mindex], empirical_dir +'/'+args.model+'_alpha_'+str(args.alpha)+'_samples_'+str(args.samples)+'_repetitions_'+str(args.repetitions)+'_idx_'+str(batch_idx)+ '.pth')

        print(str(batch_idx)+ ': '+ str(adversarial_dists[mindex]))
