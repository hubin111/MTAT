import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from core.metrics import accuracy
from core.utils import SmoothCrossEntropyLoss
from core.utils import track_bn_stats
from timm.loss import SoftTargetCrossEntropy
import torch.distributed as dist
from functools import partial
from torch.autograd import Function
import scipy.linalg

def get_mma_loss(weight):
    '''
    MMA regularization in PyTorch
    :param weight: parameter of a layer in model, out_features *　in_features
    :return: mma loss
    '''

    # for convolutional layers, flatten
    if weight.dim() > 2:
        weight = weight.view(weight.size(0), -1)

    # computing cosine similarity: dot product of normalized weight vectors
    weight_ = F.normalize(weight, p=2, dim=1)
    cosine = torch.matmul(weight_, weight_.t())

    # make sure that the diagnonal elements cannot be selected
    cosine = cosine - 2. * torch.diag(torch.diag(cosine))

    # maxmize the minimum angle
    loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()

    return loss
class MatrixSquareRoot(Function):
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input

def torch_cov(input_vec:torch.tensor):
    x = input_vec- torch.mean(input_vec,axis=0)
    cov_matrix = torch.matmul(x.T, x) / (x.shape[0]-1)
    return cov_matrix

class AT(nn.Module):
	def __init__(self, p):
		super(AT, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am
class CenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # print("x", x.size())
        # print(self.centers.t().size())
        distmat.addmm_(1, -2, x, self.centers.t())
        classes = torch.arange(self.num_classes).long().cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss

def dlr_loss_targeted(x, y, y_target):
    x_sorted, ind_sorted = x.sort(dim=1)
    u = torch.arange(x.shape[0])

    return -(x[u, y] - x[u, y_target]) / (x_sorted[:, -1] - .5 * (
            x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)

class RBFLogits(nn.Module):
    def __init__(self, feature_dim, class_num, scale, gamma):
        super(RBFLogits, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.weight = nn.Parameter( torch.FloatTensor(class_num, feature_dim))
        self.bias = nn.Parameter(torch.FloatTensor(class_num))
        self.scale = scale
        self.gamma = gamma
        nn.init.xavier_uniform_(self.weight)
    def forward(self, feat, label):
        diff = torch.unsqueeze(self.weight, dim=0) - torch.unsqueeze(feat, dim=1)
        diff = torch.mul(diff, diff)
        metric = torch.sum(diff, dim=-1)
        kernal_metric = torch.exp(-1.0 * metric / self.gamma)
        if self.training:
            train_logits = self.scale * kernal_metric
            # ###
            # Add some codes to modify logits, e.g. margin, temperature and etc.
            # ###
            return train_logits
        else:
            test_logits = self.scale * kernal_metric
            return test_logits

class EQLv2(nn.Module):
    def __init__(self, use_sigmoid=True, reduction='mean', class_weight=None, loss_weight=1.0,
                 num_classes=10,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 vis_grad=False,
                 test_with_obj=True):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True
        # cfg for eqlv2
        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(self.num_classes).cuda())
        self.register_buffer('neg_grad', torch.zeros(self.num_classes).cuda())
        # At the beginning of training, we set a high value (eg. 100)
        # for the initial gradient ratio so that the weight for pos gradients and neg gradients are 1.
        self.register_buffer('pos_neg', (torch.ones(self.num_classes) * 100).cuda())

        self.test_with_obj = test_with_obj

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)

    def forward(self, cls_score, label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()
        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target
        # print("target", target)

        target = expand_label(cls_score, label)
        # print("target", target)
        pos_w, neg_w = self.get_weight(cls_score)
        weight = pos_w * target + neg_w * (1 - target)
        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target, reduction='none')
        cls_loss = torch.sum(cls_loss * weight) / self.n_i
        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())
        return self.loss_weight * cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def get_activation(self, cls_score):
        cls_score = torch.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        if self.test_with_obj:
            cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)
        # do not collect grad for objectiveness branch [:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0)
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)
        # dist.all_reduce(pos_grad)
        # dist.all_reduce(neg_grad)
        self.pos_grad += pos_grad.cuda()
        self.neg_grad += neg_grad.cuda()
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)


    def get_weight(self, cls_score):
        neg_w = torch.cat([self.map_func(self.pos_neg.cuda())])
        pos_w = 1 + self.alpha * (1 - neg_w)
        neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
        pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w


def similarity_loss(f_s, f_t):
    bsz = f_s.shape[0]
    f_s = f_s.view(bsz, -1)
    f_t = f_t.view(bsz, -1)
    G_s = torch.mm(f_s, torch.t(f_s))
    G_s = G_s / G_s.norm(2)
    G_s = torch.nn.functional.normalize(G_s)
    G_t = torch.mm(f_t, torch.t(f_t))
    G_t = G_t / G_t.norm(2)
    G_t = torch.nn.functional.normalize(G_t)
    G_diff = G_t - G_s
    loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
    return loss

class AT(nn.Module):
    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p
    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))
        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2,3), keepdim=True)
        am = torch.div(am, norm+eps)
        return am

class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = t * mask1
    t2 = t * mask2
    # rt = torch.cat([t1, t2], dim=1)
    return t1, t2

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()




def trades_loss(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, 
                attack='linf-pgd', label_smoothing=0.1, use_cutmix=False):
    """
    TRADES training (Zhang et al, 2019).
    """
  
    criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    criterions = SoftTargetCrossEntropy()
    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)
    
    x_adv = x_natural.detach() +  torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if use_cutmix: # CutMix
        p_natural = y
    else:
        p_natural = F.softmax(model(x_natural), dim=1)
        p_natural = p_natural.detach()
        target = F.one_hot(y, num_classes=10).float()
    
    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                # loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), target)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    elif attack == 'l2-pgd':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    model.train()
    track_bn_stats(model, True)
  
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    optimizer.zero_grad()
    # calculate robust loss
    logits_natural = model(x_natural)
    logits_adv = model(x_adv)

    if use_cutmix: # CutMix
        loss_natural = criterion_kl(F.log_softmax(logits_natural, dim=1), y)
    else:
        loss_natural = criterion_ce(logits_natural, y)

    logits_an = 0.5 * logits_natural + 0.5 * logits_adv
    loss_an = criterion_ce(logits_an, y)

    for i in range(10):
        lam = np.random.beta(1, 1)
        logits_ana = lam * logits_natural + (1 - lam) * logits_adv
        loss_an += criterion_ce(logits_ana, y)
    loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))
    loss = loss_natural + beta * loss_robust
    
    if use_cutmix: # CutMix
        batch_metrics = {'loss': loss.item(), 'clean_acc': -1, 
                     'adversarial_acc': -1}
    else:
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits_natural.detach()), 
                     'adversarial_acc': accuracy(y, logits_adv.detach())}
        
    return loss, batch_metrics

def trades_loss_LSE(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, 
                attack='linf-pgd', label_smoothing=0.1, clip_value=0, use_cutmix=False, num_classes=10):
    """
    TRADES training (Zhang et al, 2019).
    """
    criterion_cent = CenterLoss(num_classes=num_classes, feat_dim=640)
    criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction='sum')
    criterion = torch.nn.SmoothL1Loss(reduction='mean')
    criterion_hu = nn.MultiMarginLoss(margin=0.3)
    criterions = SoftTargetCrossEntropy()
    criterionAT = AT(2.0)
    MMD = MMDLoss()
    EQ = EQLv2()
    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)
    x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if use_cutmix: # CutMix
        p_natural = y
    else:
        p_natural = model(x_natural)
        p_natural = F.softmax(p_natural, dim=1)
        p_natural = p_natural.detach()
        pred = torch.argmax(p_natural, dim=1)
        m = torch.eq(pred, y)
        s = (1-m.float()).to(dtype=torch.bool)
        #target = (1 - num_classes * label_smoothing / (num_classes - 1)) * F.one_hot(y, num_classes=num_classes) + label_smoothing
        target = F.one_hot(y, num_classes=10).float()
        hu = 0.5*target + 0.5*p_natural
        # print("m", torch.sum(m) / (len(y)))
        # x_advv = x_adv[m]
        # x_advs = x_adv[s]
        # predv = torch.argmax(p_natural, dim=1)
        # c = (predv==y).squeeze()
        # correct = list(0. for i in range(10))
        # total = list(0. for i in range(10))

        # for i in range(batch_size):
        #     y1 = y[i]
        #     correct[y1] += c[i].item()
        #     total[y1] += 1
        # print("ren",np.around(100*np.array(correct)/np.array(total)))

    
    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            # x_advv.requires_grad_()
            # x_advs.requires_grad_()
            x_adv.requires_grad_()
            # output_advv = F.softmax(model(x_advv), dim=1)
            # output_advs = F.softmax(model(x_advs), dim=1)
            output_adv1 = model(x_adv)

            # output_adv2 = F.softmax(output_adv1[:, [0, 1, 2, 3, 7]], dim=1).cuda()
            # output_adv3 = F.softmax(output_adv1[:, [8, 9, 4, 5, 6]], dim=1).cuda()
            # adv2 = torch.cat((output_adv2[:, [0, 1, 2, 3]], output_adv3[:, [2, 3, 4]]), dim=-1)
            # adv3 = torch.cat((adv2, output_adv2[:, [4]]), dim=-1)
            # output_adv = torch.cat((adv3, output_adv3[:, [0, 1]]), dim=-1)
            output_adv = F.softmax(output_adv1, dim=1)
            # predv = torch.argmax(output_advv, dim=1)
            # predvs = torch.argmax(output_adv, dim=1)
            # c = (predvs == y).squeeze()
            # correct = list(0. for i in range(10))
            # total = list(0. for i in range(10))
            #
            # for i in range(batch_size):
            #     y1 = y[i]*(2/(hubin + 1))
            #     correct[y1] += c[i].item()
            #     total[y1] += 1
            # rensh = torch.Tensor(np.array(correct) / np.array(total)).cuda()
            # mv = torch.eq(predv, y[m])
            # mv = torch.eq(predvs, y)
            # sv = (1 - mv.float()).to(dtype=torch.bool)

            # guo, ke = output_adv.topk(4, dim=1)
            # hua_maxy = torch.gather(output_adv, 1, torch.unsqueeze(y, 1))
            # hua_max = torch.gather(output_adv, 1, torch.unsqueeze(ke[:, 0], 1))
            # print("mv", torch.sum(mv) / (len(y)))
            # print("mvs", torch.sum(mvs)/(len(y)))
            # ccc = (mv.float() - m.float()) > 0
            # bbb = (m.float() - mv.float()) > 0
            # aaa = (m.float() - bbb.float()) > 0
            # eee = ccc + bbb + aaa
            # hefei1 = output_adv * target
            # hubei1 = output_adv * (1 - target)
            # noo1 = hu * target
            # non1 = hu * (1 - target)
            with torch.enable_grad():
                # n1, n2 = cat_mask(output_adv, gt_mask, other_mask)
                # hefei = output_adv*target
                # hubei = output_adv*(1-target)
                # loss_lse = torch.sum(3*(hubei - (1 - target)) ** 2 + (hefei - target) ** 2, dim=-1).mean()
                # loss_lse = criterion_kl(F.log_softmax(output_adv, dim=-1), F.softmax(target, dim=-1))
                # logits_naturalm, indices_naturalm = output_adv.topk(4, dim=1)
                # logits_maxy = torch.gather(output_adv, 1, torch.unsqueeze(y, 1))
                # logits_max = torch.gather(output_adv, 1, torch.unsqueeze(indices_naturalm[:, 0], 1))
                # logits_cimax = torch.gather(output_adv, 1, torch.unsqueeze(indices_naturalm[:, 1], 1))
                loss_lse = torch.sum((output_adv - hu) ** 2)
                # loss_lse = torch.sum((output_adv - p_natural) ** 2)
                # loss_lse = 100 * torch.sum((hua_maxy - hua_max) ** 2)
                # loss_lse1 = torch.sum(torch.mv((output_adv - hu) ** 2, 1/rensh))
                # loss_lse += torch.sum(F.cosine_similarity(output_adv, hu))


                # loss_lse = 2 * torch.sum((hefei1 - noo1) ** 2) + torch.sum((hubei1 - non1) ** 2)
                # print("hubin",100 * torch.sum((hua_maxy - hua_max) ** 2))
                # print(loss_lse)
                # loss_lse = torch.sum((output_adv - hu) ** 2) + (1/batch_size) * torch.sum((logits_maxy - logits_max) ** 2)
                # loss_lse = torch.sum((output_adv - p_natural) ** 2, dim=-1)
                # print(torch.sum((logits_maxy - 1) ** 2))
                # loss_lse = criterion_ce(output_adv, y)
                # loss_lse = torch.sum((output_advv - p_natural[m]) ** 2)
                # loss_lses = torch.sum((output_advs - p_natural[s]) ** 2)
                # loss_lse = torch.sum((output_adv[m] - p_natural[m]) ** 2) + 2 * torch.sum((output_adv[s] - p_natural[s]) ** 2)
                # loss_lse = to.sum((n1 - tt1) ** 2, dim=-1).mean() + torch.sum((n2 - yy2) ** 2, dim=-1).mean()
                # loss_lse = MMD(source=output_adv, target=target)
                # loss_lse = criterion_kl(output_adv, target)
            grad = torch.autograd.grad(loss_lse, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

            # gradv = torch.autograd.grad(loss_lse, [x_advv])[0]
            # grads = torch.autograd.grad(loss_lses, [x_advs])[0]
            # x_advv = x_advv.detach() + step_size * torch.sign(gradv.detach())
            # x_advv = torch.min(torch.max(x_advv, x_natural[m] - epsilon), x_natural[m] + epsilon)
            # x_advv = torch.clamp(x_advv, 0.0, 1.0)

            # x_advs = x_advs.detach() + step_size * torch.sign(grads.detach())
            # x_advs = torch.min(torch.max(x_advs, x_natural[s] - epsilon), x_natural[s] + epsilon)
            # x_advs = torch.clamp(x_advs, 0.0, 1.0)

    elif attack == 'l2-pgd':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                output_adv = F.softmax(model(adv), dim=1)
                loss = (-1) * torch.sum((output_adv - p_natural) ** 2)
            loss.backward()

            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    model.train()
    track_bn_stats(model, True)
  
    # x_advv = Variable(torch.clamp(x_advv, 0.0, 1.0), requires_grad=False)
    # x_advs = Variable(torch.clamp(x_advs, 0.0, 1.0), requires_grad=False)
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()

    if use_cutmix: # CutMix
        y_onehot = y
    else:
        y_onehot = (1 - num_classes * label_smoothing / (num_classes-1)) * F.one_hot(y, num_classes=num_classes) + label_smoothing / (num_classes-1)

    logits_natural1 = model(x_natural)
    logits_natural = F.softmax(logits_natural1, dim=-1)

    # logits_natural1_norms = torch.norm(logits_natural1, p=2, dim=-1, keepdim=True) + 1e-7
    # logits_natural11 = torch.div(logits_natural1, logits_natural1_norms)
    # logits_natural11 = F.softmax(logits_natural11, dim=-1)

    # print("chen", logits_natural11[0])
    # print("jie", logits_natural[0])

    # logits_natural2 = F.softmax(logits_natural1[:, [2, 3, 4, 5]], dim=1).cuda()
    # logits_natural3 = F.softmax(logits_natural1[:, [0, 1, 6, 7, 8, 9]], dim=1).cuda()
    # # logits_natural3 = F.softmax(logits_natural1[:, [8, 9, 4, 5, 6]], dim=1).cuda()
    # natural2 = torch.cat((logits_natural3[:, [0, 1]], logits_natural2[:, [0, 1, 2, 3]]), dim=-1)
    # natural3 = torch.cat((natural2, logits_natural3[:, [2, 3, 4, 5]]), dim=-1)
    # logits_natural = torch.cat((natural3, logits_natural3[:, [0, 1]]), dim=-1)
    # logits_natural = natural3
    # logits_natural = F.softmax(model(x_natural), dim=1)
    logits_adv1 = model(x_adv)
    logits_adv = F.softmax(logits_adv1, dim=1)
    # logits_adv = wujian(logits_adv1)

    # logits_adv1_norms = torch.norm(logits_adv1, p=2, dim=-1, keepdim=True) + 1e-7
    # logits_adv11 = torch.div(logits_adv1, logits_adv1_norms)
    # logits_adv11 = F.softmax(logits_adv11, dim=1)
    # print(logits_adv11[0])

    # logits_adv2 = F.softmax(logits_adv1[:, [0, 1, 2, 3, 7]], dim=1).cuda()
    # logits_adv3 = F.softmax(logits_adv1[:, [8, 9, 4, 5, 6]], dim=1).cuda()
    # adv2 = torch.cat((logits_adv2[:, [0, 1, 2, 3]], logits_adv3[:, [2, 3, 4]]), dim=-1)
    # adv3 = torch.cat((adv2, logits_adv2[:, [4]]), dim=-1)
    # logits_adv = torch.cat((adv3, logits_adv3[:, [0, 1]]), dim=-1
    # logits_natural1_norms = torch.norm(logits_natural1, p=2, dim=-1, keepdim=True) + 1e-7
    # logits_natural11 = torch.div(logits_natural1, logits_natural1_norms)
    #
    # logits_adv1_norms = torch.norm(logits_adv1, p=2, dim=-1, keepdim=True) + 1e-7
    # logits_adv11 = torch.div(logits_adv1, logits_adv1_norms)
    #adv1 torch.

    # print("loss_s", loss_s)
    # print("logits_natural-mean", torch.sum(torch.mean(logits_natural1[mv], dim=1)[0:3]))
    # print("logits_natural-mean1", torch.sum(torch.mean(logits_natural1[sv], dim=1)[0:3]))
    # print("logits_natural-var", torch.sum(torch.var(logits_natural1[mv], dim=1)[0:3]))
    # print("logits_natural-var1", torch.sum(torch.var(logits_natural1[sv], dim=1)[0:3]))
    # print("logits_adv-mean", torch.sum(torch.mean(logits_adv1[mv], dim=1)[0:3]))
    # print("logits_adv-mean1", torch.sum(torch.mean(logits_adv1[sv], dim=1)[0:3]))
    # print("logits_adv-var", torch.sum(torch.var(logits_adv1[mv], dim=1)))
    # print("logits_adv-var1", torch.sum(torch.var(logits_adv1[sv], dim=1)))
    # logits_adv = F.softmax(model(x_advv), dim=1)
    # logits_advs = F.softmax(model(x_advs), dim=1)
    # loss_naturall = criterion_ce(logits_natural, y)
    # logits_naturall = torch.softmax(logits_natural, dim=1)
    # logits_naturalm, indices_naturalm = logits_natural.topk(4, dim=1)
    # y_target = indices_naturalm[:, 0]
    # print("hefei",indices_naturalm[:, 1])
    # print("linkeda", y)
    # logits_natural_max = torch.gather(logits_natural[sv], 1, torch.unsqueeze(y[sv], 1))
    # # logits_natural_max = torch.gather(logits_natural, 1, torch.unsqueeze(indices_naturalm[:, 0], 1))
    # logits_natural_cimax = torch.gather(logits_natural, 1, torch.unsqueeze(indices_naturalm[:, 1], 1))
    # logits_natural_cicimax = torch.gather(logits_natural[sv], 1, torch.unsqueeze(indices_naturalm[:, 2], 1))
    # logits_natural_cicicimax = torch.gather(logits_natural[sv], 1, torch.unsqueeze(indices_naturalm[:, 3], 1))

    # loss2 = 0.05 / torch.exp((1.0 / batch_size) * torch.sum(distance_na - 0.6))
    # logits_advm, indices_advm = logits_adv.topk(4, dim=1)
    # y_target_adv = indices_advm[:, 0]
    # logits_adv_maxy = torch.gather(logits_adv, 1, torch.unsqueeze(y, 1))
    # logits_adv_max = torch.gather(logits_adv, 1, torch.unsqueeze(indices_advm[:, 0], 1))
    # logits_adv_cimax = torch.gather(logits_adv, 1, torch.unsqueeze(indices_advm[:, 1], 1))
    # logits_adv_cicimax = torch.gather(logits_adv, 1, torch.unsqueeze(indices_advm[:, 2], 1))
    # logits_adv_cicicimax = torch.gather(logits_adv, 1, torch.unsqueeze(indices_advm[:, 3], 1))
    # logits_adv_cimax_m = logits_adv_cimax + logits_adv_cicimax + logits_adv_cicicimax
    # xu = logits_adv_max > logits_adv_cimax_m
    # jin = (1-xu.float()).to(dtype=torch.bool)
    # logits_adv_cimax[jin] = logits_adv_cimax_m[jin]
    # logits_adv[:, torch.unsqueeze(indices_advm[:, 1], 1)] = logits_adv_cimax


    # gt_mask = _get_gt_mask(logits_natural, y)
    # other_mask = _get_other_mask(logits_natural, y)
    # t1, t2 = cat_mask(logits_natural, gt_mask, other_mask)
    # s1, s2 = cat_mask(logits_adv, gt_mask, other_mask)
    # y1, y2 = cat_mask(y_onehot, gt_mask, other_mask)

    # logits_an = 0.75 * logits_natural + 0.25 * logits_adv
    # loss_an = criterion_ce(logits_an, y)
    # y01, y02 = cat_mask(y_onehot, gt_mask2, other_mask2)
    # an1, an2 = cat_mask(logits_an, gt_mask, other_mask)
    # loss_an = torch.sum((an1 - y1) ** 2, dim=-1).mean() + 3 * torch.sum((an2 - y2) ** 2, dim=-1).mean()

    # for i in range(6):
    #     lam = np.random.beta(1, 1)
    #     logits_ana = 0.75 * lam * logits_natural + 0.25 * (1 - lam) * logits_adv
    #     loss_an += criterion_ce(logits_ana, y)

        # ana1, ana2 = cat_mask(logits_an, gt_mask, other_mask)
        # y11, y22 = cat_mask(y_onehot, gt_mask3, other_mask3)
        # loss_an += torch.sum((ana1 - y1) ** 2, dim=-1).mean() + 3 * torch.sum((ana2 - y2) ** 2, dim=-1).mean()
        # loss_an += torch.sum((logits_ana - y_onehot) ** 2, dim=-1).mean()
    # loss_natural = torch.sum((logits_natural - yes) ** 2, dim=-1).mean()

    # loss_natural = torch.sum((logits_natural[m] - y_onehot[m]) ** 2, dim=-1).mean() + torch.sum((logits_natural[s] - y_onehot[s]) ** 2, dim=-1).mean()
    # print("loss_robustt",torch.sum((logits_natural - y_onehot) ** 2, dim=-1).mean())
    # print("loss2", torch.sum((distance_na - 0.6) ** 2, dim=-1).mean())
    # logits_bn = (logits_natural + y_onehot)/2
    # d1, d2 = cat_mask(logits_adv, gt_mask, other_mask)
    # c = (predv==y).squeeze()
    # correct = list(0. for i in range(10))
    # total = list(0. for i in range(10))
    #
    # for i in range(batch_size):
    #     y1 = y[i]
    #     correct[y1] += c[i].item()
    #     total[y1] += 1
    # print("sheng",np.around(100*np.array(correct)/np.array(total)))
    # pred = torch.argmax(logits_natural, dim=1)
    # c = (pred == y).squeeze()
    # correct = list(0. for i in range(10))
    # total = list(0. for i in range(10))
    # for i in range(batch_size):
    #     y1 = y[i]
    #     correct[y1] += c[i].item()
    #     total[y1] += 1
    # hubei = torch.Tensor(np.array(correct) / np.array(total)).cuda()
    # print("hubei", hubei)
    predv = torch.argmax(logits_adv, dim=1)
    mv = torch.eq(predv, y)
    sv = (1 - mv.float()).to(dtype=torch.bool)

    # predv11 = torch.argmax(logits_adv11, dim=1)
    # mv11 = torch.eq(predv11, y)
    # sv11 = (1 - mv11.float()).to(dtype=torch.bool)
    # hu = torch.abs(s.float()-sv.float()).to(dtype=torch.bool)
    # print(torch.sum(s))
    # print(torch.sum(sv))
    # yes = 0.8 *logits_natural + 0.2 * logits_adv
    # loss_natural = criterion_ce(yes[mv], y[mv]) + 2*criterion_ce(yes[sv], y[sv])
    # p = logits_natural_max[sv] > 0.65
    # q = logits_natural_max[mv] > 0.65
    # pp = logits_adv_max[sv]-logits_adv_cimax[sv] > 0.2
    # qq = logits_adv_max[mv]-logits_adv_cimax[mv] > 0.2
    # a = torch.unsqueeze(indices_naturalm[:, 1], 1).eq(torch.unsqueeze(indices_advm[:, 0], 1))#cimax
    # c = torch.unsqueeze(indices_naturalm[:, 2], 1).eq(torch.unsqueeze(indices_advm[:, 0], 1))#cicimax
    # d = torch.unsqueeze(indices_naturalm[:, 3], 1).eq(torch.unsqueeze(indices_advm[:, 0], 1))#cicicimax
    # cc = torch.squeeze(c, dim=1)
    # dd = torch.squeeze(d, dim=1)
    # aa = torch.squeeze(a, dim=1)
    # loss_aa = torch.sum((logits_adv_max[aa] - logits_natural_cimax[aa]) ** 2, dim=-1).mean()
    # loss_cc = torch.sum((logits_adv_max[cc] - logits_natural_cicimax[cc]) ** 2, dim=-1).mean()
    # loss_dd = torch.sum((logits_adv_max[dd] - logits_natural_cicicimax[dd]) ** 2, dim=-1).mean()
    # loss_hu = loss_aa + loss_cc + loss_dd
    # aaa = torch.eq(aa, s)
    # aaa = (1 - aaa.float()).to(dtype=torch.bool)
    # aaa = torch.eq(aaa, cc)
    # aaa = (1 - aaa.float()).to(dtype=torch.bool)
    # aaa = torch.eq(aaa, dd)
    # aaa = (1 - aaa.float()).to(dtype=torch.bool)
    # # print(torch.sum(aaa))
    # b = torch.unsqueeze(indices_naturalm[:, 0], 1).eq(torch.unsqueeze(indices_advm[:, 0], 1))
    # bb = torch.squeeze(b, dim=1)
    # bbb = torch.eq(bb, s)
    # aaa = (1 - aaa.float()).to(dtype=torch.bool)
    # print(bbb)
    # ccc = torch.eq(bbb, aaa)
    # print(torch.sum(a)+torch.sum(c)+torch.sum(s))
    # print("ccc",torch.sum(ccc))
    # print("s", torch.sum(p)/torch.sum(sv))
    # print("m", torch.sum(q)/torch.sum(mv))
    # print("sv", torch.sum(pp) / torch.sum(sv))
    # print("mv", torch.sum(qq) / torch.sum(mv))
    # print("msv", batch_size/torch.sum(mv))

    # ccc = (mv.float() - m.float()) > 0
    # bbb = (m.float() - mv.float()) > 0
    # aaa = (m.float() - bbb.float()) > 0
    # eee = ccc + bbb + aaa

    # ddd = (1-ddd).to(dtype=torch.bool)
    # print(torch.sum(aaa))
    # print(torch.sum(bbb))
    # print(torch.sum(ccc))
    # print(batch_size-torch.sum(aaa)-torch.sum(bbb)-torch.sum(ccc))
    # hub = torch.ones_like(1-logits_adv_max[sv])
    # loss_hu = criterion(1-logits_adv_max[sv], 0.1*hub)
    # print("niu", 512-torch.sum(aaa)-torch.sum(bbb)-torch.sum(ccc))
    # eee = (s.float() - sv.float()) > 0
    # ddd = (s.float() - eee.float()) > 0
    # print(loss_hu)
    yes = y_onehot
    # yes = 0.8 * y_onehot + 0.2 * logits_adv
    # yes11 = 0.8 * y_onehot + 0.2 * logits_adv11
    # yes = (y_onehot - logits_natural)*(1-y_onehot) + y_onehot
    # loss_natural = torch.sum((logits_natural[mv] - yes[mv]) ** 2), dim=-1).mean() + 2*torch.sum((logits_natural[sv] - yes[sv]) ** 2, dim=-1).mean()
    # loss_natural = criterion(logits_natural[mv], yes[mv]) + criterion(logits_natural[sv], yes[sv])
    # loss_natural = torch.sum((logits_natural[mv] - yes2[mv]) ** 2, dim=-1).mean() + criterion_ce(logits_natural[sv], y[sv]) + loss_hu
    # loss_natural = (10.0 / batch_size) * criterion_kl(F.log_softmax(logits_natural1, dim=1), F.softmax(yes, dim=1))
    # print("niu", (logits_natural[mv] - yes[mv]).size())

    # loss_natural = (10.0 / batch_size) * criterion_kl(F.log_softmax(logits_natural1, dim=1), F.softmax(yes, dim=1))
    # loss_natural = (5.0 / torch.sum(mv)) * criterion_kl(F.log_softmax(logits_natural1[mv], dim=1), F.softmax(yes[mv], dim=1)) + (10.0 / torch.sum(sv)) * criterion_kl(F.log_softmax(logits_natural1[sv], dim=1), F.softmax(yes[sv], dim=1))
    # loss_robust = MMD(source=logits_adv, target= y_onehot).mean()
    # loss_natural = torch.cosine_similarity(logits_natural[mv], yes[mv]).mean() + 2 * torch.cosine_similarity(logits_natural[sv], yes[sv]).mean()
    # hubin = 5 * torch.sum((logits_adv[bbb]-y_onehot[bbb]) ** 2, dim=-1).mean()
    # no = 0.1 * y_onehot + 0.9 * logits_natural
    no = 0.5 * y_onehot + 0.5 * logits_natural
    # no = (y_onehot - logits_adv) * (1 - y_onehot) + y_onehot*no
    # print("loss_robustt", (logits_adv[mv] - no[mv]) ** 2)
    # print("loss2", (logits_adv[sv] - no[sv]) ** 2)

    # cnto = 0
    # cntp = 0
    # cntj = 0
    # zong = torch.zeros(logits_adv.size()).cuda()
    # cha = torch.zeros(logits_adv.size()).cuda()
    # sha = torch.zeros(logits_adv.size()).cuda()
    # c = (predv == y).squeeze()
    # correct = list(0. for i in range(100))
    # total = list(0. for i in range(100))
    # for i in range(batch_size):
    #     y1 = y[i]
    #     if y1 == 3:
    #         zong[i] = logits_adv[i]
    #         cntj += 1
    #     if y1 == 3 and c[i] == True:
    #         cha[i] = logits_adv[i]
    #         cnto += 1
    #     if y1 == 3 and c[i] == False:
    #         sha[i] = logits_adv[i]
    #         cntp += 1
    #
    #     correct[y1] += c[i].item()
    #     total[y1] += 1
    # cap = torch.sum(cha, dim=0)/cnto
    # lock = torch.sum(sha, dim=0)/cntp
    # ctrl = torch.sum(zong, dim=0) / cntj
    # # print("acc", cnto/cntj)
    # # print("ctrl", ctrl)
    # # print("cap", cap)
    # # print("lock", lock)
    # hubin = torch.Tensor(np.array(correct) / np.array(total)).cuda()
    # print(hubin)


    # y1 = torch.argmax(y_onehot, dim=1)
    # hf = logits_natural

    # y1 = torch.argmax(logits_adv[mv], dim=1)
    # hf = logits_adv[mv]
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
    #
    #
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
    # hubin = np.array(np.around(100*np.array(correct)/np.array(total)))
    # print("correct", correct)
    # print(np.array(correct)+np.array(h))
    # huhe = torch.Tensor(np.array(correct)/np.array(total)).cuda()
    # hubin = F.softmax(huhe)
    # print(huhe)
    # print("h", hubin)
    #loss_natural = criterion_hu(logits_natural, y)
    # loss_natural = criterion_ce(logits_natural1, y)
    # loss_natural += 0.5 * criterion_ce(logits_natural11, y)
    # t = torch.eye(10).cuda()
    # t = torch.reshape(t, (hbba.size())).cuda()
    # loss_natural = torch.sum((hbba-t) ** 2, dim=-1).mean()
    # o = torch.zeros_like(y_onehot)
    # for i in range(batch_size):
    #     if torch.argmax(y_onehot, dim=1)[i] == 0:
    #         o[i, 0] = 1
    #         o[i, 1] = 1
    #         o[i, 3] = 1
    #         o[i, 7] = 1
    #     if torch.argmax(y_onehot, dim=1)[i] == 1:
    #         o[i, 0] = 1
    #         o[i, 1] = 1
    #         o[i, 2] = 1
    #         o[i, 3] = 1
    #         o[i, 7] = 1
    #     if torch.argmax(y_onehot, dim=1)[i] == 2:
    #         o[i, 0] = 1
    #         o[i, 1] = 1
    #         o[i, 2] = 1
    #         o[i, 3] = 1
    #         o[i, 7] = 1
    #     if torch.argmax(y_onehot, dim=1)[i] == 3:
    #         o[i, 0] = 1
    #         o[i, 1] = 1
    #         o[i, 2] = 1
    #         o[i, 3] = 1
    #         o[i, 7] = 1
    #     if torch.argmax(y_onehot, dim=1)[i] == 7:
    #         o[i, 0] = 1
    #         o[i, 1] = 1
    #         o[i, 2] = 1
    #         o[i, 3] = 1
    #         o[i, 7] = 1
    # print(o)
    # print(y_onehot)
    # loss_natural = 1000*similarity_loss(logits_natural, logits_adv)
    # print("loss_natural", loss_natural)
    # loss_natural = F.cross_entropy(logits_natural, y, weight=2/(hubin+1))
    # pt = torch.exp(-loss_natural)
    # loss_natural = (1-pt)**2*loss_natural
    # print("loss_natural", loss_natural)
    # criterion_ce1 = nn.BCELoss()
    # lin = criterion_ce1(no, o)
    # print("lin", lin)
    # hefei = logits_natural*y_onehot
    # hubei = logits_natural*(1-y_onehot)
    # yeso = yes * y_onehot
    # yesn = yes * (1 - y_onehot)
    # loss_natural = torch.mv(3*(hubei - yeso) ** 2 + (hefei - yesn) ** 2, 2/(hubin + 1)).mean()
    # shen = logits_natural[:, [0, 8]]
    # yes = yes*(2/(hubin + 1))
    # for name, m in model.named_modules():
    #     # 'name' can be used to exclude some specified layers
    #     if isinstance(m, (nn.Linear, nn.Conv2d)):
    #         mma_loss = get_mma_loss(m.weight)
    # tmp1 = torch.argsort(logits_natural, dim=1)[:, -2:]
    # new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    # true_probs = torch.gather(logits_natural[mv], 1, (y[mv].unsqueeze(1)).long()).squeeze()
    # false_probs = torch.gather(logits_natural[sv], 1, (y[sv].unsqueeze(1)).long()).squeeze()
    loss_natural = torch.sum((logits_natural[mv] - yes[mv]) ** 2, dim=-1).mean() + 2*torch.sum((logits_natural[sv] - yes[sv]) ** 2, dim=-1).mean()

    # loss_natural = torch.sum((logits_natural - y_onehot) ** 2, dim=-1).mean()
    # pt = torch.exp(-loss_natural)
    # loss_natural = (1-pt)**2*loss_natural
                   # + F.nll_loss(torch.log(1.0001 - logits_natural + 1e-12), new_y)
    # loss_natural = (1.0 / len(mv)) * torch.sum(torch.sum((logits_natural[mv] - yes[mv]) ** 2, dim=-1) * (1.0000001 - true_probs)) \
    #                 + 2 * (1.0 / len(sv)) * torch.sum(torch.sum((logits_natural[sv] - yes[sv]) ** 2, dim=-1) * (1.0000001 - false_probs))
    # loss_natural += 0.5*torch.sum((logits_natural11[mv11] - yes11[mv11]) ** 2, dim=-1).mean() + torch.sum((logits_natural11[sv11] - yes11[sv11]) ** 2, dim=-1).mean()
    # loss_natural += 0.5 * mma_loss
    # Ec_natural = -torch.logsumexp(logits_natural1, dim=1)
    # Ec_adv = -torch.logsumexp(logits_adv1, dim=1)
    # loss_i = 0.1*batch_size * torch.pow(F.relu(Ec_natural - Ec_adv), 2).mean()
    # loss_natural += (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_natural, dim=1), F.softmax(yes, dim=1))
    # print("loss_natural1", (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_natural, dim=1), F.softmax(yes, dim=1)))
    # loss_natural = torch.sum((logits_natural[mv]*y_onehot[mv] - yes[mv]*y_onehot[mv]) ** 2, dim=-1).mean() + 2*torch.sum((logits_natural[sv]*y_onehot[sv] - yes[sv]*y_onehot[sv]) ** 2, dim=-1).mean()
    # loss_natural += torch.sum((logits_natural[mv] * (1-y_onehot[mv]) - 0) ** 2, dim=-1).mean() + 2 * torch.sum(
    #     (logits_natural[sv] * (1-y_onehot[sv]) - 0) ** 2, dim=-1).mean()
    # zhu, xiang = logits_natural.topk(4, dim=1)
    # yuan_maxy = torch.gather(logits_natural, 1, torch.unsqueeze(y, 1))
    # yuan_max = torch.gather(logits_natural, 1, torch.unsqueeze(xiang[:, 0], 1))
    # loss_natural = torch.mv((logits_natural[mv] - yes[mv]) ** 2, 2/(hubin + 1)).mean() + 2 * torch.mv((logits_natural[sv] - yes[sv]) ** 2, 2/(hubin + 1)).mean()

    # mu1 = torch.mean(logits_natural, dim=0)
    # mu2 = torch.mean(yes, dim=0)
    # sigma1 = torch_cov(logits_natural.T)
    # sigma2 = torch_cov(yes.T)
    # ssdiff = torch.sum((mu1 - mu2) ** 2.0)
    # # calculate sqrt of product between cov
    # sqrtm = MatrixSquareRoot.apply
    # covmean = sqrtm(torch.mm(sigma1, sigma2))
    # loss_natural = ssdiff + torch.trace(sigma1 + sigma2 - 2.0 * covmean)
    # loss_natural = ssdiff + torch.trace(sigma1 + sigma2)

    # loss_natural = torch.mv((logits_natural[mv] - yes[mv]) ** 2,  (logits_adv - no) ** 2).mean()
    # loss_natural += 0.01*criterion_cent(ppp, y)
    # loss_natural = torch.mv((logits_natural - yes) ** 2, 2 / (hubin + 1)).mean()
    # loss_natural += (1.0 / len(sv)) * torch.sum((yuan_maxy[sv] - 1) ** 2).mean()
    # loss_natural += (2.0/batch_size)*torch.sum(F.cosine_similarity(logits_natural[sv], yes[sv]))
    # print("loss_natural1", loss_natural1)
    # print("loss_naturalrrrrr", 2 * torch.mv((logits_natural[sv] - y_onehot[sv]) ** 2, 1/hubin).mean())
    # loss_natural = (100 / batch_size) * criterion_kl(F.log_softmax(logits_natural, dim=1), F.softmax(y_onehot, dim=1))
    # # t = torch.eye(10).cuda()
    # loss_natural += 0.1 * EQ(logits_natural, y.cuda())
    # hbb = torch.cat((hb0, hb1, hb2, hb3, hb4, hb5, hb6, hb7, hb8, hb9), dim=0).reshape(10, 10).cuda()
    # hbba = torch.cat((hb0a, hb1a, hb2a, hb3a, hb4a, hb5a, hb6a, hb7a, hb8a, hb9a), dim=0).reshape(10, 10).cuda()
    # loss_natural += torch.sum((hbb - t) ** 2, dim=-1).mean()
    # loss_natural += torch.sum((hbba - t) ** 2, dim=-1).mean()
    # print("loss_natural", loss_natural)
    # loss_natural += torch.sum((logits_natural[sv][:, [0, 8]] - y_onehot[sv][:, [0, 8]]) ** 2, dim=-1).mean()
    # loss_natural += torch.sum((logits_natural[sv][:, [1, 9]] - y_onehot[sv][:, [1, 9]]) ** 2, dim=-1).mean()
    # loss_natural += 2 * torch.sum((logits_natural[sv][:, [2, 4]] - y_onehot[sv][:, [2, 4]]) ** 2, dim=-1).mean()
    # loss_natural += 2 * torch.sum((logits_natural[sv][:, [3, 5]] - y_onehot[sv][:, [3, 5]]) ** 2, dim=-1).mean()
    # loss_natural += torch.sum((logits_natural[sv][:, [6, 7]] - y_onehot[sv][:, [6, 7]]) ** 2, dim=-1meiy).mean()
    # adv1 = torch.sum(logits_adv[:, [2]], dim=-1).unsqueeze(1)
    # adv2 = torch.sum(logits_adv[:, [4]], dim=-1).unsqueeze(1)
    # adv = torch.cat((adv1, adv2), dim=-1).cuda()
    # adv1y = torch.sum(y_onehot[:, [2]], dim=-1).unsqueeze(1)
    # adv2y = torch.sum(y_onehot[:, [4]], dim=-1).unsqueeze(1)
    # advy = torch.cat((adv1y, adv2y), dim=1).cuda()
    #
    # adv11 = torch.sum(logits_adv[:, [3]], dim=-1).unsqueeze(1)
    # adv22 = torch.sum(logits_adv[:, [5]], dim=-1).unsqueeze(1)
    # adv1 = torch.cat((adv11, adv22), dim=-1).cuda()
    # adv1y1 = torch.sum(y_onehot[:, [3]], dim=-1).unsqueeze(1)
    # adv2y1 = torch.sum(y_onehot[:, [5]], dim=-1).unsqueeze(1)
    # advy1 = torch.cat((adv1y1, adv2y1), dim=1).cuda()
    # criterion_ce1 = nn.BCELoss()
    # lin = 0.1*criterion_ce1(adv, advy) + 0.1*criterion_ce1(adv1, advy1)
    # print("lin", lin)
    # hefei = logits_adv*y_onehot
    # hubei = logits_adv*(1-y_onehot)
    # noo = no*y_onehot
    # non = no*(1-y_onehot)
    # loss_robust1 = 2*torch.sum((hefei - noo) ** 2, dim=-1).mean()
    # print("loss_robust1 ", loss_robust1)
    # loss_robust = 10*torch.sum((hubei - non) ** 2, dim=-1).mean()
    # print("loss_robust ", loss_robust)
    # loss_robust = criterion_ce(logits_adv, y)
    # no = no*(2/(hubin + 1))
    # loss_robust = torch.sum((logits_adv - no) ** 2, dim=-1).mean()
    # loss_robust += (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(no, dim=1))
    # print("loss_robust ", (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(no, dim=1)))

    # logitss = F.cosine_similarity(logits_natural11, logits_adv11,  dim=-1)
    # print("logitss", logitss.size())
    # logits = torch.mm(logits_natural1, logits_adv1.t())
    # labelss = torch.arange(batch_size).long().cuda()
    # loss_i = F.cross_entropy(logits, labelss)
    # loss_t = F.cross_entropy(logits.t(), labelss)
    # loss_s = (loss_i + loss_t) / 2
    # loss_robust += torch.sum((logits_adv*logits_natural - y_onehot) ** 2, dim=-1).mean()
    # loss_robust += 10 * torch.sum((logits_adv11 - logits_natural11) ** 2, dim=-1).mean()
    # chen, tian = logits_adv.topk(4, dim=1)
    # yu_maxy = torch.gather(logits_adv, 1, torch.unsqueeze(y, 1))
    # yu_max = torch.gather(logits_adv, 1, torch.unsqueeze(tian[:, 0], 1))
    # loss_robust = torch.mv((logits_adv - no) ** 2, 2/(hubin + 1)).mean()
    # loss_robust += 500*criterionAT(qqq, ppp)

    # print(loss_robust1)
    # print(0.1*torch.sum((qqq - ppp) ** 2, dim=-1).mean())
    # loss_robust += (1.0 / batch_size) * torch.sum((yu_maxy - 1) ** 2)
    # loss_robust += 0.001 * torch.sum(dlr_loss_targeted(logits_adv, y, y_target_adv), dim=0)
    # print("loss_robust1 ", loss_robust1)
    # loss_robust = torch.mv(2*(hefei - noo) ** 2 + (hubei - non) ** 2, 2/(hubin + 1)).mean()
    loss_robust = torch.sum((logits_adv - no) ** 2, dim=-1).mean()

    # loss_robust = torch.sum((logits_adv - logits_natural) ** 2, dim=-1).mean()
    # loss_robust = torch.mv((logits_adv - no) ** 2, 2/(hubin+1)).mean()
    # print("loss_robust ", loss_robust)
    # print("loss_robusteeeee", torch.mv((logits_adv[mv] - y_onehot[mv]) ** 2, 1 / hubin).mean())
    # print("loss_robustrrrrr", torch.mv((logits_adv[sv] - y_onehot[sv]) ** 2, 1 / hubin).mean())
                #  torch.sum((adv - advy) ** 2, dim=-1).mean()
    # loss_robust += 3*torch.sum((logits_adv[:, [0, 1, 2, 3, 7]] - no[:, [0, 1, 2, 3, 7]]) ** 2, dim=-1).mean()
    # loss_robust += torch.sum((logits_adv[:, [8, 9, 4, 5, 6]] - no[:, [8, 9, 4, 5, 6]]) ** 2, dim=-1).mean()
    # loss_robust = 0.06 * EQ(logits_adv, y.cuda())+ 0.02 * EQ(logits_adv, y.cuda())
    # loss_robust += torch.sum((logits_adv[sv][:, [0, 8]] - y_onehot[sv][:, [0, 8]]) ** 2, dim=-1).mean()
    # loss_robust += torch.sum((logits_adv[sv][:, [1, 9]] - y_onehot[sv][:, [1, 9]]) ** 2, dim=-1).mean()
    # loss_robust += 2 * torch.sum((logits_adv[sv][:, [2, 4]] - y_onehot[sv][:, [2, 4]]) ** 2, dim=-1).mean()
    # loss_robust += 2 * torch.sum((logits_adv[sv][:, [3, 5]] - y_onehot[sv][:, [3, 5]]) ** 2, dim=-1).mean()
    # loss_robust += torch.sum((logits_adv[sv][:, [6, 7]] - y_onehot[sv][:, [6, 7]]) ** 2, dim=-1).mean()

    # loss_natural += torch.sum((logits_natural_max -1) ** 2, dim=-1).mean() + torch.sum((logits_natural_cimax) ** 2, dim=-1).mean()
    # print(criterionAT(act_na, act_adv).detach()*5000)
    # print(torch.sum((logits_natural_max -1) ** 2, dim=-1).mean() + torch.sum((logits_natural_cimax) ** 2, dim=-1).mean())
    # loss_natural += torch.sum((logits_natural[mv] - y_onehot[mv]) ** 2, dim=-1).mean() + 2*torch.sum((logits_natural[sv] - y_onehot[sv]) ** 2, dim=-1).mean()
    # loss_natural = torch.sum((logits_natural - y_onehot) ** 2, dim=-1).mean()

    # cnt = 1
    # for i in range(6):
    #     lam = np.random.beta(1, 1)
    #     if lam >= 0.5:
    #         logits_bn = lam * logits_natural + (1 - lam) * logits_adv
    #         loss_natural += criterion_ce(logits_bn, y)
    #         cnt += 1
    #     else:
    #         loss_natural = loss_natural
    # loss_natural = (1 / cnt) * loss_natural
    # loss_natural = torch.sum((logits_natural[mv]* y_onehot[mv] - yes[mv]) ** 2, dim=-1).mean() + 2 * torch.sum(
    #     (logits_natural[sv] - yes[sv]) ** 2, dim=-1).mean()


    # hh = (logits_adv[mv]* y_onehot[mv]).mean()
    # bb = (logits_adv[sv]* y_onehot[sv]).mean()
    # print("hubin", hh)
    # print("hefei", bb)
    #loss_robust = 100*similarity_loss(logits_adv, y_onehot)
    # print("loss_robust",loss_robust)
    # loss_robust = torch.sum((logits_adv - logits_natural) ** 2, dim=-1).mean() + (1/batch_size) * torch.sum((logits_adv_maxy - logits_adv_max) ** 2).mean()
    # # loss_robust = torch.sum((logits_adv - no) ** 2, dim=-1).mean() + (1/batch_size) * torch.sum((logits_adv_maxy - logits_adv_max) ** 2).mean()

    # loss_robust = torch.sum(2*((hubei1 - (1 - logits_natural*(1-y_onehot))) ** 2) + (hefei1 - logits_natural*y_onehot) ** 2, dim=-1).mean()
    # loss_robust = 3*torch.sum((hubei - no*(1-y_onehot)) ** 2, dim=-1).mean()
    # loss_robust1 = 2*torch.sum((hefei - noo) ** 2, dim=-1).mean() + torch.sum((hubei - non) ** 2, dim=-1).mean()
    # print("loss_robust1 ", loss_robust1)
    # ptt = torch.exp(-loss_robust)
    # loss_robust = (1-ptt)**2*loss_robust
    # loss_robust = criterion_ce(logits_adv, y)
    # print(loss_robust)
    # for i in range(6):
    #     lam = np.random.beta(1, 1)
    #     logits_no = lam * logits_natural + (1 - lam) * y_onehot
    #     loss_robust += torch.sum((logits_adv - logits_no) ** 2, dim=-1).mean()
    # loss_robust = 1/7 * loss_robust
    # loss_robust = torch.sum((logits_adv - no) ** 2, dim=-1).mean()
    # loss_robust = criterion_ce(logits_adv, y)

    # print("bin", bin)
    # loss_robust = criterion_ce(logits_adv, y)
    # distance_ana = torch.sum((1-logits_natural_max[mv] + logits_natural_cimax[mv]) ** 2, dim=-1).mean()
    # distance_na = torch.sum((1-logits_natural_max[sv] + logits_natural_cimax[sv]) ** 2, dim=-1).mean()
    # loss_ls = distance_na*0.5
    # loss_robust = criterion(logits_adv, logits_natural)
    # loss_robust = torch.sum((logits_adv[mv] - logits_natural[mv]) ** 2, dim=-1).mean() + torch.sum((logits_adv[sv] - y_onehot[sv]) ** 2, dim=-1).mean()
    # for i in range(6):
    # lam = np.random.beta(1, 1)
    # logits_bn = lam * logits_natural + (1 - lam) * y_onehot
    # loss_robust += torch.sum((logits_natural - logits_bn) ** 2, dim=-1).mean() + torch.sum((logits_adv - logits_bn) ** 2, dim=-1).mean()

    #loss_robust = criterion_ce(logits_adv, y)
    # loss_robust1 = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))
    loss_robust = F.relu(loss_robust - clip_value) # clip loss value
    # loss = (1.0 / 3.0) * loss_an + loss_natural.mean() + beta * loss_robust.mean() + loss2 + 6.0 * loss_robustt.mean()
    loss = loss_natural + beta * loss_robust
    # loss = loss_natural + loss_robust
    if use_cutmix: # CutMix
        batch_metrics = {'loss': loss.item(), 'clean_acc': -1,
                     'adversarial_acc': -1}
    else:
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits_natural.detach()), 
                     'adversarial_acc': accuracy(y, logits_adv.detach())}
        
    return loss, batch_metrics