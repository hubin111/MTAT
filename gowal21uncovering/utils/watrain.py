import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attacks import create_attack
from core.attacks import CWLoss
from core.metrics import accuracy
from core.models import create_model

from core.utils import ctx_noparamgrad_and_eval
from core.utils import Trainer
from core.utils import set_bn_momentum
from core.utils import seed

from .trades import trades_loss, trades_loss_LSE
from .cutmix import cutmix
import copy

from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import torchvision

def show(img):
    img = img
    npimg = img
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
    plt.savefig("hubin.png")


def wujian(out):
    # out = F.softmax(out, dim=1)
    tmp1 = torch.argsort(out, dim=1)[:, -2:]
    jim = torch.gather(out, 1, tmp1[:, -1].unsqueeze(dim=1)) - torch.gather(out, 1, tmp1[:, -2].unsqueeze(dim=1)) > 0.1
    for i in range(len(out)):
        if jim[i] == False:
            # print(out[i])
            # temp = copy.deepcopy(out[i][tmp1[:, -1][i]].detach())
            # tempp = copy.deepcopy(out[i][tmp1[:, -2][i]].detach())
            temp = out[i][tmp1[:, -1][i]].clone()
            tempp = out[i][tmp1[:, -2][i]].clone()
            # print(tempp)
            out[i][tmp1[:, -2][i]] += temp-tempp
            out[i][tmp1[:, -1][i]] += tempp-temp
            # print(out[i])
        else:
            out[i] = out[i]
    return out


def junke(out):
    predv = torch.argmax(out, dim=1)
    huhe = torch.zeros_like(predv)
    adv1 = torch.sum(out[:, [0, 1, 2, 3, 7]], dim=-1).unsqueeze(1)
    adv2 = torch.sum(out[:, [8, 9, 4, 5, 6]], dim=-1).unsqueeze(1)
    adv = torch.cat((adv1, adv2), dim=-1)
    predv1 = torch.argmax(adv, dim=1).float()
    predv0 = (1 - predv1).float().to(dtype=torch.bool)
    predv1 = predv1.to(dtype=torch.bool)
    # print(predv1)
    predv2 = torch.argmax(out[predv0][:, [0, 1, 2, 3, 7]], dim=1)
    for i in range(len(predv2)):
        if predv2[i] == 0:
            predv2[i] = 0
        if predv2[i] == 1:
            predv2[i] = 1
        if predv2[i] == 2:
            predv2[i] = 2
        if predv2[i] == 3:
            predv2[i] = 3
        if predv2[i] == 4:
            predv2[i] = 7
    # huhe[1 - predv1] = predv2
    # print("predv2", predv2.size())
    predv3 = torch.argmax(out[predv1][:, [8, 9, 4, 5, 6]], dim=1)
    for i in range(len(predv3)):
        if predv3[i] == 0:
            predv3[i] = 8
        if predv3[i] == 1:
            predv3[i] = 9
        if predv3[i] == 2:
            predv3[i] = 4
        if predv3[i] == 3:
            predv3[i] = 5
        if predv3[i] == 4:
            predv3[i] = 6
    # huhe[predv1] = predv3
    # print("predv3",predv3.size())
    i = 0
    t = 0
    for j in range(len(huhe)):
        if predv1[j] == 0:
            huhe[j] = predv2[i]
            i = i + 1
        if predv1[j] == 1:
            huhe[j] = predv3[t]
            t = t + 1
    # hefei = torch.eq(huhe.cpu(), y.cpu()).sum().float() / float(y.size(0))
    # print(hefei)
    return huhe

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WATrainer(Trainer):
    """
    Helper class for training a deep neural network with model weight averaging (identical to Gowal et al, 2020).
    Arguments:
        info (dict): dataset information.
        args (dict): input arguments.
    """
    def __init__(self, info, args):
        super(WATrainer, self).__init__(info, args)

        seed(args.seed)
        self.wa_model = copy.deepcopy(self.model)
        self.eval_attack = create_attack(self.wa_model, CWLoss, args.attack, args.attack_eps, 4*args.attack_iter,
                                         args.attack_step)
        num_samples = 50000 if 'cifar' in self.params.data else 73257
        num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
        if self.params.data in ['cifar10', 'cifar10s', 'svhn', 'svhns']:
            self.num_classes = 10
        elif self.params.data in ['cifar100', 'cifar100s']:
            self.num_classes = 100
        elif self.params.data == 'tiny-imagenet':
            self.num_classes = 200
        self.update_steps = int(np.floor(num_samples/self.params.batch_size) + 1)
        self.warmup_steps = 0.025 * self.params.num_adv_epochs * self.update_steps


    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and schedulers.
        """
        def group_weight(model):
            group_decay = []
            group_no_decay = []
            for n, p in model.named_parameters():
                if 'batchnorm' in n:
                    group_no_decay.append(p)
                else:
                    group_decay.append(p)
            assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
            groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
            return groups

        self.optimizer = torch.optim.SGD(group_weight(self.model), lr=self.params.lr, weight_decay=self.params.weight_decay,
                                         momentum=0.9, nesterov=self.params.nesterov)
        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs)


    def train(self, dataloader, epoch=0, adversarial=False, verbose=True):
        """
        Run one epoch of training.
        """
        metrics = pd.DataFrame()
        self.model.train()

        update_iter = 0
        for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
            global_step = (epoch - 1) * self.update_steps + update_iter
            if global_step == 0:
                # make BN running mean and variance init same as Haiku
                set_bn_momentum(self.model, momentum=1.0)
            elif global_step == 1:
                set_bn_momentum(self.model, momentum=0.01)
            update_iter += 1

            x, y = data
            if self.params.CutMix:
                x_all, y_all = torch.tensor([]), torch.tensor([])
                for i in range(4): # 128 x 4 = 512 or 256 x 4 = 1024
                    x_tmp, y_tmp = x.detach(), y.detach()
                    x_tmp, y_tmp = cutmix(x_tmp, y_tmp, alpha=1.0, beta=1.0, num_classes=self.num_classes)
                    x_all = torch.cat((x_all, x_tmp), dim=0)
                    y_all = torch.cat((y_all, y_tmp), dim=0)
                x, y = x_all.to(device), y_all.to(device)
            else:
                x, y = x.to(device), y.to(device)

            if adversarial:
                if self.params.beta is not None and self.params.mart:
                    loss, batch_metrics = self.mart_loss(x, y, beta=self.params.beta)
                elif self.params.beta is not None and self.params.LSE:
                    loss, batch_metrics = self.trades_loss_LSE(x, y, beta=self.params.beta)
                elif self.params.beta is not None:
                    loss, batch_metrics = self.trades_loss(x, y, beta=self.params.beta)
                else:
                    loss, batch_metrics = self.adversarial_loss(x, y)
            else:
                loss, batch_metrics = self.standard_loss(x, y)

            loss.backward()
            if self.params.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)
            self.optimizer.step()
            if self.params.scheduler in ['cyclic']:
                self.scheduler.step()

            global_step = (epoch - 1) * self.update_steps + update_iter
            ema_update(self.wa_model, self.model, global_step, decay_rate=self.params.tau,
                       warmup_steps=self.warmup_steps, dynamic_decay=True)
            metrics = metrics.append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)

        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()

        update_bn(self.wa_model, self.model)
        return dict(metrics.mean())


    def trades_loss(self, x, y, beta):
        """
        TRADES training.
        """
        loss, batch_metrics = trades_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step,
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter,
                                          beta=beta, attack=self.params.attack, label_smoothing=self.params.ls,
                                          use_cutmix=self.params.CutMix)
        return loss, batch_metrics

    def trades_loss_LSE(self, x, y, beta):
        """
        TRADES training with LSE loss.
        """

        loss, batch_metrics = trades_loss_LSE(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                          beta=beta, attack=self.params.attack, label_smoothing=self.params.ls,
                                          clip_value=self.params.clip_value,
                                          use_cutmix=self.params.CutMix,
                                          num_classes=self.num_classes)
        return loss, batch_metrics


    
    def eval(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        hefei = 0.0
        self.wa_model.eval()
        correct = list(0. for i in range(10))
        total = list(0. for i in range(10))
        # cnt = 1
        # hubin = np.array(np.around(100 * np.array(list(0. for i in range(10)))))
       # images, labels =iter(dataloader).next()
        #show(torchvision.utils.make_grid(images[2]))
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            if adversarial:
                with ctx_noparamgrad_and_eval(self.wa_model):
                    x_adv, _ = self.eval_attack.perturb(x, y)
                # y_onehot = F.one_hot(y, num_classes=10).float()
                out = self.wa_model(x_adv)

                # out0 = self.wa_model(x)
                # out = 0.4 * y_onehot + 0.6 * out0
                # outt = 0.4 * y_onehot + 0.6 * out1

                # logits_advm, indices_advm = out.topk(4, dim=1)
                # # logits_adv_maxy = torch.gather(logits_adv, 1, torch.unsqueeze(y, 1))
                # logits_adv_max = torch.gather(out, 1, torch.unsqueeze(indices_advm[:, 0], 1))
                # logits_adv_cimax = torch.gather(out, 1, torch.unsqueeze(indices_advm[:, 1], 1))
                # logits_adv_cicimax = torch.gather(out, 1, torch.unsqueeze(indices_advm[:, 2], 1))
                # logits_adv_cicicimax = torch.gather(out, 1, torch.unsqueeze(indices_advm[:, 3], 1))
                # logits_adv_cimax_m = logits_adv_cimax + logits_adv_cicimax
                # xu = logits_adv_max > logits_adv_cimax_m
                # # print(xu)
                # jin = (1 - xu.float()).to(dtype=torch.bool)
                # logits_adv_cimax[jin] = logits_adv_cimax_m[jin]
                # out[:, torch.unsqueeze(indices_advm[:, 1], 1)] = logits_adv_cimax

                # y_onehot = F.one_hot(y, num_classes=10).float()
                predv = torch.argmax(out, dim=1)
                mv = torch.eq(predv, y)
                sv = (1 - mv.float()).to(dtype=torch.bool)
                # predv = torch.argmax(out, dim=1)
                # huhe = torch.zeros_like(predv)
                # adv1 = torch.sum(out[:, [0, 1, 2, 3, 7]], dim=-1).unsqueeze(1)
                # adv2 = torch.sum(out[:, [8, 9, 4, 5, 6]], dim=-1).unsqueeze(1)
                # adv = torch.cat((adv1, adv2), dim=-1)
                # predv1 = torch.argmax(adv, dim=1).float()
                # predv0 = (1 - predv1).float().to(dtype=torch.bool)
                # predv1 = predv1.to(dtype=torch.bool)
                # # print(predv1)
                # predv2 = torch.argmax(out[predv0][:, [0, 1, 2, 3, 7]], dim=1)
                # for i in range(len(predv2)):
                #     if predv2[i] == 0:
                #         predv2[i] = 0
                #     if predv2[i] == 1:
                #         predv2[i] = 1
                #     if predv2[i] == 2:
                #         predv2[i] = 2
                #     if predv2[i] == 3:
                #         predv2[i] = 3
                #     if predv2[i] == 4:
                #         predv2[i] = 7
                # # huhe[1 - predv1] = predv2
                # # print("predv2", predv2.size())
                # predv3 = torch.argmax(out[predv1][:, [8, 9, 4, 5, 6]], dim=1)
                # for i in range(len(predv3)):
                #     if predv3[i] == 0:
                #         predv3[i] = 8
                #     if predv3[i] == 1:
                #         predv3[i] = 9
                #     if predv3[i] == 2:
                #         predv3[i] = 4
                #     if predv3[i] == 3:
                #         predv3[i] = 5
                #     if predv3[i] == 4:
                #         predv3[i] = 6
                # # huhe[predv1] = predv3
                # # print("predv3",predv3.size())
                # i = 0
                # t = 0
                # for j in range(len(huhe)):
                #     if predv1[j] == 0:
                #         huhe[j] = predv2[i]
                #         i = i + 1
                #     if predv1[j] == 1:
                #         huhe[j] = predv3[t]
                #         t = t + 1


                # print(hefei)
                # huhe = junke(out)
                # hefei = torch.eq(huhe.cpu(), y.cpu()).sum().float() / float(y.size(0))
                # print(huhe.size())

                # hh = (out[mv] * y_onehot[mv]).mean()
                # bb = (out[sv] * y_onehot[sv]).mean()
                # print("hubin", hh)
                # print("hefei", bb)
                # c = (huhe == y).squeeze()
                c = (predv == y).squeeze()
                for i in range(len(y)):
                    y1 = y[i]
                    correct[y1] += c[i].item()
                    total[y1] += 1
                # print(np.around(100*np.array(correct) / np.array(total)))
                # h = np.array(np.array(correct) / np.array(total))
                # hubin += h
                # cnt += 1
            else:
                out = self.wa_model(x)

                # img = np.transpose(x[10].cpu().numpy(), (1, 2, 0))
                # plt.imshow(img)
                # # cv2.imshow("", img)
                # plt.savefig("hubin.png")
                # plt.show()
                # x1 = cv2.imread("/root/SCORE/hubin.png")
                # # x1 = cv2.cvtColor(x1, cv2.COLOR_RGB2BGR)
                # # x1 = img
                # print("x1", x1.shape)
                # heat = out1[10].data.cpu().numpy()  # 将tensor格式的feature map转为numpy格式
                # # heat = np.squeeze(heat, 0)  # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除
                # for i in range(heat.shape[0]):
                #     heats = heat[i, :, :]
                #     # print(heats.shape)
                #     cam = heats - np.min(heats)
                #     cam_img = cam / np.max(cam)
                #     heats = np.uint8(255 * cam_img)
                #     heatmap = cv2.resize(heats, (x1.shape[1], x1.shape[0]))
                #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                #     print("map", heatmap.shape)
                #     superimposed_img = heatmap * 0.4 + x1
                #     if i ==1:
                #         cv2.imwrite("%d.jpg" % (4), superimposed_img)
                #         cv2.imwrite("%d.jpg" % (5), x1)

            # out = F.softmax(out, dim=1)
            # tmp1 = torch.argsort(out, dim=1)[:, -2:]
            # jim = torch.gather(out, 1, tmp1[:, -1].unsqueeze(dim=1))-torch.gather(out, 1, tmp1[:, -2].unsqueeze(dim=1)) > 0.05
            # new_y = torch.where(jim.squeeze(), tmp1[:, -1], tmp1[:, -2]).squeeze()
            # acc += ((new_y == y).sum().float()/float(y.size(0))).cpu()
            # out = wujian(out)
            acc += accuracy(y, out)
        # hubin = hubin/cnt
        # for i in range(10):
        #     print("Accuracy of %5s:%2d %%" % (y[i], 100* correct[i]/total[i]))
        acc /= len(dataloader)
        return acc





    def save_model(self, path):
        """
        Save model weights.
        """
        torch.save({
            'model_state_dict': self.wa_model.state_dict(), 
            'unaveraged_model_state_dict': self.model.state_dict()
        }, path)

    
    def load_model(self, path):
        """
        Load model weights.
        """
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        # hubin = checkpoint['model_state_dict']
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    

def ema_update(wa_model, model, global_step, decay_rate=0.995, warmup_steps=0, dynamic_decay=True):
    """
    Exponential model weight averaging update.
    """
    factor = int(global_step >= warmup_steps)
    if dynamic_decay:
        delta = global_step - warmup_steps
        decay = min(decay_rate, (1. + delta) / (10. + delta)) if 10. + delta != 0 else decay_rate
    else:
        decay = decay_rate
    decay *= factor
    
    for p_swa, p_model in zip(wa_model.parameters(), model.parameters()):
        p_swa.data *= decay
        p_swa.data += p_model.data * (1 - decay)


@torch.no_grad()
def update_bn(avg_model, model):
    """
    Update batch normalization layers.
    """
    avg_model.eval()
    model.eval()
    for module1, module2 in zip(avg_model.modules(), model.modules()):
        if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
            module1.running_mean = module2.running_mean
            module1.running_var = module2.running_var
            module1.num_batches_tracked = module2.num_batches_tracked
