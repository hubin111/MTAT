import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from core.attacks import create_attack
from core.metrics import accuracy
from core.models import create_model

from .context import ctx_noparamgrad_and_eval
from .utils import seed

from .mart import mart_loss
from .rst import CosineLR
from .trades import trades_loss
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCHEDULERS = ['cyclic', 'step', 'cosine', 'cosinew']

def tsne_plot(targets, outputs):
    print('generating t-SNE plot...')
    if not os.path.exists('/root/SCORE/results'):
        os.makedirs('/root/SCORE/results')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(n_components=2, random_state=0)
    tsne_output = tsne.fit_transform(outputs)
    x_min, x_max = tsne_output.min(0), tsne_output.max(0)
    tsne_output = (tsne_output-x_min)/(x_max - x_min)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['Classes'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='Classes',
        palette=sns.color_palette("hls", 10),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(os.path.join('/root/SCORE/results','tsne.png'), bbox_inches='tight')
    print('done!')


class Trainer(object):
    """
    Helper class for training a deep neural network.
    Arguments:
        info (dict): dataset information.
        args (dict): input arguments.
    """
    def __init__(self, info, args):
        super(Trainer, self).__init__()
        
        seed(args.seed)
        self.model = create_model(args.model, args.normalize, info, device)

        self.params = args
        self.criterion = nn.CrossEntropyLoss()
        self.init_optimizer(self.params.num_adv_epochs)
        
        if self.params.pretrained_file is not None:
            #self.load_model(os.path.join(self.params.log_dir, self.params.pretrained_file, 'weights-best.pt'))
            # self.load_model(os.path.join(self.params.log_dir, self.params.pretrained_file, 'WRN-28-10_cifar10.pt'))
            # self.load_model(os.path.join(self.params.log_dir, self.params.pretrained_file, 'WRN-34-10_cifar10.pth')) #/WRN-34-10_cifar10.pth
            self.load_model(os.path.join(self.params.log_dir, self.params.pretrained_file, 'Pre.pth'))
        self.attack, self.eval_attack = self.init_attack(self.model, self.criterion, self.params.attack, self.params.attack_eps, 
                                                         self.params.attack_iter, self.params.attack_step)

#/root/SCORE/trained_models/WRN34-10Swish_cifar100
    @staticmethod
    def init_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step):
        """
        Initialize adversary.
        """
        attack = create_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step, rand_init_type='uniform')
        if attack_type in ['linf-pgd', 'l2-pgd']:
            eval_attack = create_attack(model, criterion, attack_type, attack_eps, 2*attack_iter, attack_step)
        elif attack_type in ['fgsm', 'linf-df']:
            eval_attack = create_attack(model, criterion, 'linf-pgd', 8/255, 20, 2/255)
        elif attack_type in ['fgm', 'l2-df']:
            eval_attack = create_attack(model, criterion, 'l2-pgd', 128/255, 20, 15/255)
        return attack,  eval_attack
    
    
    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and scheduler.
        """
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay, 
                                         momentum=0.9, nesterov=self.params.nesterov)
        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs)
    
        
    def init_scheduler(self, num_epochs):
        """
        Initialize scheduler.
        """
        if self.params.scheduler == 'cyclic':
            num_samples = 50000 if 'cifar10' in self.params.data else 73257
            num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
            update_steps = int(np.floor(num_samples/self.params.batch_size) + 1)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.25,
                                                                 steps_per_epoch=update_steps, epochs=int(num_epochs))
        elif self.params.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1, milestones=[100, 105])    
        elif self.params.scheduler == 'cosine':
            self.scheduler = CosineLR(self.optimizer, max_lr=self.params.lr, epochs=int(num_epochs))
        elif self.params.scheduler == 'cosinew':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.025, 
                                                                 total_steps=int(num_epochs))
        else:
            self.scheduler = None
    
    
    def train(self, dataloader, epoch=0, adversarial=False, verbose=True):
        """
        Run one epoch of training.
        """
        metrics = pd.DataFrame()
        self.model.train()
        
        for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
            x, y = data
            x, y = x.to(device), y.to(device)
            
            if adversarial:
                if self.params.beta is not None and self.params.mart:
                    loss, batch_metrics = self.mart_loss(x, y, beta=self.params.beta)
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
            
            metrics = metrics.append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
        
        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()
        return dict(metrics.mean())
    
    
    def standard_loss(self, x, y):
        """
        Standard training.
        """
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = self.criterion(out, y)
        
        preds = out.detach()
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, preds)}
        return loss, batch_metrics
    
    
    def adversarial_loss(self, x, y):
        """
        Adversarial training (Madry et al, 2017).
        """
        with ctx_noparamgrad_and_eval(self.model):
            x_adv, _ = self.attack.perturb(x, y)
        
        self.optimizer.zero_grad()
        if self.params.keep_clean:
            x_adv = torch.cat((x, x_adv), dim=0)
            y_adv = torch.cat((y, y), dim=0)
        else:
            y_adv = y
        out = self.model(x_adv)
        loss = self.criterion(out, y_adv)
        
        preds = out.detach()
        batch_metrics = {'loss': loss.item()}
        if self.params.keep_clean:
            preds_clean, preds_adv = preds[:len(x)], preds[len(x):]
            batch_metrics.update({'clean_acc': accuracy(y, preds_clean), 'adversarial_acc': accuracy(y, preds_adv)})
        else:
            batch_metrics.update({'adversarial_acc': accuracy(y, preds)})    
        return loss, batch_metrics
    
    
    def trades_loss(self, x, y, beta):
        """
        TRADES training.
        """
        loss, batch_metrics = trades_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                          beta=beta, attack=self.params.attack)
        return loss, batch_metrics  

    
    def mart_loss(self, x, y, beta):
        """
        MART training.
        """
        loss, batch_metrics = mart_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                        epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                        beta=beta, attack=self.params.attack)
        return loss, batch_metrics  
    
    
    def eval1(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        self.model.eval()
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        correct = list(0. for i in range(10))
        total = list(0. for i in range(10))
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.model):
                    x_adv, _ = self.eval_attack.perturb(x, y)            
                out = self.model(x_adv)
                predv = torch.argmax(out, dim=1)
                c = (predv == y).squeeze()
                for i in range(len(x)):
                    y1 = y[i]
                    correct[y1] += c[i].item()
                    total[y1] += 1
            else:
                out = self.model(x)

                tmp1 = torch.argsort(out, dim=1)[:, -2:]
                jim = torch.gather(F.softmax(out, dim=1), 1, tmp1[:, -1].unsqueeze(dim=1)) - torch.gather(
                    F.softmax(out, dim=1), 1, tmp1[:, -2].unsqueeze(dim=1)) > 0.5
                jim = jim.squeeze()
                with ctx_noparamgrad_and_eval(self.model):
                    x_adv, _ = self.eval_attack.perturb(x, y)
                out1 = self.model(x_adv)

                # predv = torch.argmax(out1, dim=1)
                # mv = torch.eq(predv, y)
                # sv = (1 - mv.float()).to(dtype=torch.bool)

                # predv = torch.argmax(out, dim=1)
                predv = torch.argmax(out[jim], dim=1)
                y = y[jim]

                c = (predv == y).squeeze()
                for i in range(len(x[jim])):
                    y1 = y[i]
                    correct[y1] += c[i].item()
                    total[y1] += 1
            # acc += accuracy(y, out)
        for i in range(10):
            print(total[i])
            print("Accuracy of %5s:%.2f %%" % (classes[i], 100 * correct[i]/total[i]))
        # acc /= len(dataloader)
        return acc

    def eval(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        self.model.eval()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.model):
                    x_adv, _ = self.eval_attack.perturb(x, y)
                out = self.model(x_adv)
            else:
                out = self.model(x)
            acc += accuracy(y, out)
        acc /= len(dataloader)
        return acc

    def eval2(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        self.model.eval()
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        correct = list(0. for i in range(10))
        total = list(0. for i in range(10))
        targets_list = []
        outputs_list = []
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            targets_np = y.data.cpu().numpy()
            if adversarial:
                with ctx_noparamgrad_and_eval(self.model):
                    x_adv, _ = self.eval_attack.perturb(x, y)
                out = self.model(x_adv)

                outputs_np = out.data.cpu().numpy()
                targets_list.append(targets_np[:, np.newaxis])
                outputs_list.append(outputs_np)
                predv = torch.argmax(out, dim=1)
                c = (predv == y).squeeze()
                for i in range(len(x)):
                    y1 = y[i]
                    correct[y1] += c[i].item()
                    total[y1] += 1
            else:
                out = self.model(x)

                outputs_np = out.data.cpu().numpy()
                targets_list.append(targets_np[:, np.newaxis])
                outputs_list.append(outputs_np)
                predv = torch.argmax(out, dim=1)
                c = (predv == y).squeeze()
                for i in range(len(x)):
                    y1 = y[i]
                    correct[y1] += c[i].item()
                    total[y1] += 1
            acc += accuracy(y, out)
        targets = np.concatenate(targets_list, axis=0)
        outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)
        tsne_plot(targets, outputs)
        for i in range(10):
            print(total[i])
            print("Accuracy of %5s:%.2f %%" % (classes[i], 100 * correct[i] / total[i]))
        acc /= len(dataloader)
        return acc

    def eval3(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        int = 0
        self.model.eval()
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        correct = list(0. for i in range(100))
        total = list(0. for i in range(100))
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.model):
                    x_adv, _ = self.eval_attack.perturb(x, y)
                out, out1 = self.model(x_adv)

                img = np.transpose(x[8].cpu().numpy(), (1, 2, 0))
                plt.imshow(img)
                # cv2.imshow("", img)
                plt.savefig("hubin.png")
                plt.show()
                x1 = cv2.imread("/root/SCORE/hubin.png")
                # x1 = cv2.cvtColor(x1, cv2.COLOR_RGB2BGR)
                # x1 = img
                # print("x1", x1.shape)

                heat = out1[8].data.cpu().numpy()  # 将tensor格式的feature map转为numpy格式
                # heat = np.squeeze(heat, 0)  # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除
                for i in range(heat.shape[0]):
                    heats = heat[i, :, :]
                    # print(heats.shape)
                    cam = heats - np.min(heats)
                    cam_img = cam / np.max(cam)
                    heats = np.uint8(255 * cam_img)
                    heatmap = cv2.resize(heats, (x1.shape[1], x1.shape[0]))
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    print("map", heatmap.shape)
                    superimposed_img = heatmap * 0.4 + x1
                    if i ==1:
                        cv2.imwrite("%d.jpg" % (int), superimposed_img)
                        cv2.imwrite("%d.jpg" % (int+1), x1)
                        int = int + 2

                predv = torch.argmax(out, dim=1)
                c = (predv == y).squeeze()
                for i in range(len(x)):
                    y1 = y[i]
                    correct[y1] += c[i].item()
                    total[y1] += 1
            else:
                out = self.model(x)
                predv = torch.argmax(out, dim=1)
                c = (predv == y).squeeze()
                for i in range(len(x)):
                    y1 = y[i]
                    correct[y1] += c[i].item()
                    total[y1] += 1
            acc += accuracy(y, out)
        for i in range(10):
            print(total[i])
            print("Accuracy of %5s:%.2f %%" % (classes[i], 100 * correct[i] / total[i]))
        acc /= len(dataloader)
        return acc


    def save_model(self, path):
        """
        Save model weights.
        """
        torch.save({'model_state_dict': self.model.state_dict()}, path)

    
    def load_model(self, path, load_opt=True):
        """
        Load model weights.
        """
        from collections import OrderedDict
        checkpoint = torch.load(path)
        try:
            if 'model_state_dict' not in checkpoint:
                raise RuntimeError('Model weights not found at {}.'.format(path))
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except:
            new = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = 'module.0.' + k[7:]
                new[name] = v
            self.model.load_state_dict(new)