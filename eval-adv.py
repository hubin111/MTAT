"""
Adversarial Evaluation with PGD+, CW (Margin) PGD and black box adversary.
"""

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import torch
import torch.nn as nn

from core.attacks import create_attack
from core.attacks import CWLoss

from core.data import get_data_info
from core.data import load_data

from core.models import create_model

from core.utils import ctx_noparamgrad_and_eval
from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed
from core.utils import Trainer

import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Setup

parse = parser_eval()
args = parse.parse_args()

LOG_DIR = args.log_dir + '/' + args.desc
with open(LOG_DIR+'/args0.txt', 'r') as f:
    old = json.load(f)
    args.__dict__ = dict(vars(args), **old)

if args.data in ['cifar10', 'cifar10s']:
    da = '/cifar10/'
elif args.data in ['cifar100', 'cifar100s']:
    da = '/cifar100/'
elif args.data in ['svhn', 'svhns']:
    da = '/svhn/'

#/root/SCORE/trained_models/WRN28-10Swish_cifar10s_wuqiong
DATA_DIR = args.data_dir + da

# LOG_DIR = args.log_dir + args.desc
WEIGHTS = LOG_DIR + '/weights-best.pt'
#WEIGHTS = LOG_DIR + '/WRN-28-10_cifar10.pt'
logger = Logger(LOG_DIR+'/log-adv.log')

info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.log('Using device: {}'.format(device))



# Load data

seed(args.seed)
_, _, _, test_dataloader = load_data(DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=False, 
                                     shuffle_train=False)



# Helper function

def eval_multiple_restarts(attack, model, dataloader, num_restarts=5, verbose=True):
    """
    Evaluate adversarial accuracy with multiple restarts.
    """
    model.eval()
    N = len(dataloader.dataset)
    is_correct = torch.ones(N).bool().to(device)
    criterion_kl = nn.KLDivLoss(reduction='sum')

    for i in tqdm(range(0, num_restarts), disable=not verbose):
        iter_is_correct = []
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            with ctx_noparamgrad_and_eval(model):

                x_adv, _ = attack.perturb(x, y)
            out = model(x_adv)
            iter_is_correct.extend(torch.softmax(out, dim=1).argmax(dim=1) == y)
        is_correct = torch.logical_and(is_correct, torch.BoolTensor(iter_is_correct).to(device))
    
    adv_acc = (is_correct.sum().float()/N).item()
    return adv_acc

def eval_multiple_restarts_advertorch(attack, model, dataloader, num_restarts=1, verbose=True):
    """
    Evaluate adversarial accuracy with multiple restarts (Advertorch).
    """
    model.eval()
    N = len(dataloader.dataset)
    is_correct = torch.ones(N).bool().to(device)
    for i in tqdm(range(0, num_restarts), disable=not verbose):
        iter_is_correct = []
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            with ctx_noparamgrad_and_eval(model):
                x_adv = attack.perturb(x, y)
            out = model(x_adv)
            iter_is_correct.extend(torch.softmax(out, dim=1).argmax(dim=1) == y)
        is_correct = torch.logical_and(is_correct, torch.BoolTensor(iter_is_correct).to(device))
    
    adv_acc = (is_correct.sum().float()/N).item()
    return adv_acc



# PGD Evaluation

seed(args.seed)
trainer = Trainer(info, args)
if 'tau' in args and args.tau:
    print ('Using WA model.')
print(WEIGHTS)
trainer.load_model(WEIGHTS)
trainer.model.eval()

# test_acc = trainer.eval(test_dataloader)
# logger.log('\nStandard Accuracy-\tTest: {:.2f}%.'.format(test_acc*100))



if args.wb:    
    # CW-PGD-40 Evaluation
    seed(args.seed)
    num_restarts = 1
    if args.attack in ['linf-pgd', 'linf-df', 'fgsm']:
        args.attack_iter, args.attack_step = 40, 0.01
    else:
        args.attack_iter, args.attack_step = 40, 30/255.0
    assert args.attack in ['linf-pgd', 'l2-pgd'], 'CW evaluation only supported for attack=linf-pgd or attack=l2-pgd !'
    attack = create_attack(trainer.model, CWLoss, args.attack, args.attack_eps, args.attack_iter, args.attack_step)
    logger.log('\n==== CW-PGD Evaluation. ====')
    logger.log('Attack: cw-{}.'.format(args.attack))
    logger.log('Attack Parameters: Step size: {:.3f}, Epsilon: {:.3f}, #Iterations: {}.'.format(args.attack_step, 
                                                                                                args.attack_eps, 
                                                                                                args.attack_iter))
    logger.log('Standard Accuracy-\tTest: {:2f}%.'.format(trainer.eval2(test_dataloader) * 100))
    # test_adv_acc = trainer.eval3(test_dataloader, adversarial=True)
    # logger.log('Standard Adversarial Accuracy-\tTest: {:.2f}%.'.format(test_adv_acc * 100))

    test_adv_acc1 = eval_multiple_restarts(attack, trainer.model, test_dataloader, num_restarts,  verbose=False)
    logger.log('Adversarial Accuracy-\tTest: {:.2f}%.'.format(test_adv_acc1*100))
    

    # PGD-40 (with 5 restarts) Evaluation
    seed(args.seed)
    num_restarts = 5
    if args.attack in ['linf-pgd', 'linf-df', 'fgsm']:
        args.attack_iter, args.attack_step = 40, 0.01
    else:
        args.attack_iter, args.attack_step = 40, 30/255.0
    attack = create_attack(trainer.model, trainer.criterion, args.attack, args.attack_eps, args.attack_iter, args.attack_step)
    logger.log('\n==== PGD+ Evaluation. ====')
    logger.log('Attack: {} with {} restarts.'.format(args.attack, num_restarts))
    logger.log('Attack Parameters: Step size: {:.3f}, Epsilon: {:.3f}, #Iterations: {}.'.format(args.attack_step, 
                                                                                                args.attack_eps, 
                                                                                                args.attack_iter))


    test_adv_acc2 = eval_multiple_restarts(attack, trainer.model, test_dataloader, num_restarts, verbose=True)
    logger.log('Adversarial Accuracy-\tTest: {:.2f}%.'.format(test_adv_acc2*100))



# Black Box Evaluation

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

if args.source != None:
    seed(args.seed)
    assert args.attack in ['linf-pgd', 'l2-pgd'], 'Black-box evaluation only supported for attack=linf-pgd or attack=l2-pgd!'
    if args.attack in ['linf-pgd', 'linf-df', 'fgsm']:        
        args.attack_iter, args.attack_step = 40, 0.01
    else:
        args.attack_iter, args.attack_step = 40, 30/255.0

    SRC_LOG_DIR = args.log_dir + args.source
    with open(SRC_LOG_DIR+'/args.txt', 'r') as f:
        src_args = json.load(f)
        src_args = dotdict(src_args)
    
    src_model = create_model(src_args.model, src_args.normalize, info, device)
    src_model.load_state_dict(torch.load(SRC_LOG_DIR + '/weights-best.pt')['model_state_dict'])
    src_model.eval()
    
    src_attack = create_attack(src_model, trainer.criterion, args.attack, args.attack_eps, args.attack_iter, args.attack_step)
    adv_acc = 0.0
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)
        with ctx_noparamgrad_and_eval(src_model):
            x_adv, _ = src_attack.perturb(x, y)            
        out = trainer.model(x_adv)
        adv_acc += accuracy(y, out)
    adv_acc /= len(test_dataloader)
    
    logger.log('\n==== Black-box Evaluation. ====')
    logger.log('Source Model: {}.'.format(args.source))
    logger.log('Attack: {}.'.format(args.attack))
    logger.log('Attack Parameters: Step size: {:.3f}, Epsilon: {:.3f}, #Iterations: {}.'.format(args.attack_step, 
                                                                                                   args.attack_eps, 
                                                                                                   args.attack_iter))
    logger.log('Black-box Adv. Accuracy-\tTest: {:.2f}%.'.format(adv_acc*100))
    del src_attack, src_model


logger.log('Script Completed.')


def plotmap(logits_natural, logits_adv, y, num_restarts=1, verbose=True):

    y_onehot = F.one_hot(y, num_classes=10).float()
    predv = torch.argmax(logits_adv, dim=1)
    mv = torch.eq(predv, y)
    sv = (1 - mv.float()).to(dtype=torch.bool)
    # y1 = torch.argmax(y_onehot, dim=1)
    # hf = logits_natural

    y1 = torch.argmax(logits_natural[mv], dim=1)
    hf = logits_natural[mv]


    # y1a = torch.argmax(logits_adv[sv], dim=1)
    # y1b = torch.argmax(y_onehot[sv], dim=1)

    # hfa = logits_adv[sv]
    # y1b = torch.argmax(y_onehot, dim=1)
    # hfa = logits_adv
    # hb0 = torch.zeros(hf[0].size()).cuda()
    # hb1 = torch.zeros(hf[0].size()).cuda()
    # hb2 = torch.zeros(hf[0].size()).cuda()
    # hb3 = torch.zeros(hf[0].size()).cuda()
    # hb4 = torch.zeros(hf[0].size()).cuda()
    # hb5 = torch.zeros(hf[0].size()).cuda()
    # hb6 = torch.zeros(hf[0].size()).cuda()
    # hb7 = torch.zeros(hf[0].size()).cuda()
    # hb8 = torch.zeros(hf[0].size()).cuda()
    # hb9 = torch.zeros(hf[0].size()).cuda()
    # n0,n1,n2, n3,n4, n5, n6, n7, n8, n9 = 0,0,0,0,0,0,0,0,0,0
    #
    # hb0a = torch.zeros(hfa[0].size()).cuda()
    # hb1a = torch.zeros(hfa[0].size()).cuda()
    # hb2a = torch.zeros(hfa[0].size()).cuda()
    # hb3a = torch.zeros(hfa[0].size()).cuda()
    # hb4a = torch.zeros(hfa[0].size()).cuda()
    # hb5a = torch.zeros(hfa[0].size()).cuda()
    # hb6a = torch.zeros(hfa[0].size()).cuda()
    # hb7a = torch.zeros(hfa[0].size()).cuda()
    # hb8a = torch.zeros(hfa[0].size()).cuda()
    # hb9a = torch.zeros(hfa[0].size()).cuda()
    # n0a,n1a,n2a, n3a,n4a, n5a, n6a, n7a, n8a, n9a = 0,0,0,0,0,0,0,0,0,0
    #
    #
    # for i in range(len(y1)):
    #     if y1[i] == 0:
    #         hb0 += hf[i]
    #         n0 = n0 + 1
    #     if y1[i] == 1:
    #         hb1 += hf[i]
    #         n1 = n1 + 1
    #     if y1[i] == 2:
    #         hb2 += hf[i]
    #         n2 = n2 + 1
    #     if y1[i] == 3:
    #         hb3 += hf[i]
    #         n3 = n3 + 1
    #     if y1[i] == 4:
    #         hb4 += hf[i]
    #         n4 = n4 + 1
    #     if y1[i] == 5:
    #         hb5 += hf[i]
    #         n5 = n5 + 1
    #     if y1[i] == 6:
    #         hb6 += hf[i]
    #         n6 = n6 + 1
    #     if y1[i] == 7:
    #         hb7 += hf[i]
    #         n7 = n7 + 1
    #     if y1[i] == 8:
    #         hb8 += hf[i]
    #         n8 = n8 + 1
    #     if y1[i] == 9:
    #         hb9 += hf[i]
    #         n9 = n9 + 1
    # hb0 = hb0 / n0
    # hb1 = hb1 / n1
    # hb2 = hb2 / n2
    # hb3 = hb3 / n3
    # hb4 = hb4 / n4
    # hb5 = hb5 / n5
    # hb6 = hb6 / n6
    # hb7 = hb7 / n7
    # hb8 = hb8 / n8
    # hb9 = hb9 / n9
    # for i in range(len(y1b)):
    #     if y1b[i] == 0:
    #         hb0a += hfa[i]
    #         n0a = n0a + 1
    #     if y1b[i] == 1:
    #         hb1a += hfa[i]
    #         n1a = n1a + 1
    #     if y1b[i] == 2:
    #         hb2a += hfa[i]
    #         n2a = n2a + 1
    #     if y1b[i] == 3:
    #         hb3a += hfa[i]
    #         n3a = n3a + 1
    #     if y1b[i] == 4:
    #         hb4a += hfa[i]
    #         n4a = n4a + 1
    #     if y1b[i] == 5:
    #         hb5a += hfa[i]
    #         n5a = n5a + 1
    #     if y1b[i] == 6:
    #         hb6a += hfa[i]
    #         n6a = n6a + 1
    #     if y1b[i] == 7:
    #         hb7a += hfa[i]
    #         n7a = n7a + 1
    #     if y1b[i] == 8:
    #         hb8a += hfa[i]
    #         n8a = n8a + 1
    #     if y1b[i] == 9:
    #         hb9a += hfa[i]
    #         n9a = n9a + 1
    # hb0a = hb0a / n0a
    # hb1a = hb1a / n1a
    # hb2a = hb2a / n2a
    # hb3a = hb3a / n3a
    # hb4a = hb4a / n4a
    # hb5a = hb5a / n5a
    # hb6a = hb6a / n6a
    # hb7a = hb7a / n7a
    # hb8a = hb8a / n8a
    # hb9a = hb9a / n9a

    # hbb = torch.cat((hb0, hb1, hb2, hb3, hb4, hb5, hb6, hb7, hb8, hb9), dim=0)
    # hbba = torch.cat((hb0a, hb1a, hb2a, hb3a, hb4a, hb5a, hb6a, hb7a, hb8a, hb9a), dim=0)
    # loss_natural = criterion_hu(logits_natural, y)
    # print("huuhuhuhuuhuhuh0", hbb.reshape(10,10))
    # print(n0+n1+n2+n3+n4+n5+n6+n7+n8+n9)
    # print("huuhuhuhuuhuhuh0a", hbba.reshape(10,10))
    # print(n0a+n1a+n2a+n3a+n4a+n5a+n6a+n7a+n8a+n9a)
    # print("rensheng", hbb-hbba)
    # print("huuhuhuhuuhuhuh2",hb2)



    return adv_acc

