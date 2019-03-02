import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm

from dataset import TSNDataSet
from models import TSN
from transforms import *
from opts import parser

import torch
from functools import partial
import pickle

# For reading pytorch 2 module
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

best_prec1 = 0


args = parser.parse_args()
print(args)

num_class = 400
model = TSN(num_class, args.num_segments, 'RGB',
            base_model=args.arch,
            consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn)

crop_size = model.crop_size
scale_size = model.scale_size
input_mean = model.input_mean
input_std = model.input_std
policies = model.get_optim_policies()
train_augmentation = model.get_augmentation()

#model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

if args.resume:
    if os.path.isfile(args.resume):
        print(("=> loading checkpoint '{}'".format(args.resume)))
        sd = torch.load(args.resume, map_location=lambda storage, loc: storage, pickle_module=pickle)
        base = model._modules['base_model'].state_dict()
        count = 0
        if True:
            for k,v in sd.items():
                if k not in base.keys():
                    continue;
                v = v.data
                if v.shape == base[k].shape:
                    base[k].copy_(v)
                    count += 1
        #args.start_epoch = checkpoint['epoch']
        #best_prec1 = checkpoint['best_prec1']
        #model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {}), {} modules loaded"
              .format(args.resume, count))
    else:
        print(("=> no checkpoint found at '{}'".format(args.resume)))

