"""
Evaluation with AutoAttack.
"""

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from autoattack import AutoAttack
    
from core.data import get_data_info
from core.data import load_data
from core.models import create_model

from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed



# Setup

parse = parser_eval()
args = parse.parse_args()

LOG_DIR = args.log_dir + '/' + args.desc
with open(LOG_DIR+'/args.txt', 'r') as f:
# with open(LOG_DIR + '/args-28-100.txt', 'r') as f:
    old = json.load(f)
    args.__dict__ = dict(vars(args), **old)

if args.data in ['cifar10', 'cifar10s']:
    da = '/cifar10/'
elif args.data in ['cifar100', 'cifar100s']:
    da = '/cifar100/'
elif args.data in ['svhn', 'svhns']:
    da = '/svhn/'
elif args.data in ['tiny-imagenet', 'svhns']:
    da = '/tiny-imagenet/'

DATA_DIR = args.data_dir + da
WEIGHTS = LOG_DIR + '/weights-best.pt'
# WEIGHTS = LOG_DIR + '/WRN-34-10_cifar10.pth'
# WEIGHTS = LOG_DIR + '/WRN-28-10_cifar100.pt'  WRN34-10Swish_cifar10
print(WEIGHTS)

log_path = LOG_DIR + '/log-aa.log'
logger = Logger(log_path)

info = get_data_info(DATA_DIR)
# BATCH_SIZE = args.batch_size
# BATCH_SIZE_VALIDATION = args.batch_size_validation
BATCH_SIZE = 128
BATCH_SIZE_VALIDATION = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.log('Using device: {}'.format(device))


# Load data

seed(args.seed)
_, _, train_dataloader, test_dataloader = load_data(DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=False, 
                                                    shuffle_train=False)

if args.train:
    logger.log('Evaluating on training set.')
    l = [x for (x, y) in train_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in train_dataloader]
    y_test = torch.cat(l, 0)
else:
    l = [x for (x, y) in test_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_dataloader]
    y_test = torch.cat(l, 0)



# Model
print(args.model)
model = create_model(args.model, args.normalize, info, device)
checkpoint = torch.load(WEIGHTS)
if 'tau' in args and args.tau:
    print ('Using WA model.')

from collections import OrderedDict
#model.load_state_dict(checkpoint['model_state_dict'])
try:
    if 'model_state_dict' not in checkpoint:
        raise RuntimeError('Model weights not found at {}.'.format(WEIGHTS))
    model.load_state_dict(checkpoint['model_state_dict'])
except:
    new = OrderedDict()
    for k, v in checkpoint.items():
        name = 'module.0.' + k[7:]
        new[name] = v
    model.load_state_dict(new)
model.eval()
del checkpoint



# AA Evaluation

seed(args.seed)
norm = 'Linf' if args.attack in ['fgsm', 'linf-pgd', 'linf-df'] else 'L2'
adversary = AutoAttack(model, norm=norm, eps=args.attack_eps, log_path=log_path, version=args.version, seed=args.seed)

if args.version == 'custom':
    adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
    adversary.apgd.n_restarts = 1
    adversary.apgd_targeted.n_restarts = 1

with torch.no_grad():
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=BATCH_SIZE_VALIDATION)

print ('Script Completed.')