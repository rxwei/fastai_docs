
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/10b_mixup_label_smoothing.ipynb

from exp.nb_10 import *

from torch.distributions.beta import Beta

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps:float=0.1, reduction='mean'):
        super().__init__()
        self.eps,self.reduction = eps,reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        losses = -log_preds.sum(dim=-1) * self.eps/c
        losses += (1-self.eps) * F.nll_loss(log_preds, target, reduction='none')
        if self.reduction == 'mean':  return losses.mean()
        elif self.reduction == 'sum': return losses.sum()
        else:                         return losses

class NoneReduce():
    def __init__(self, loss_func):
        self.loss_func,self.old_red = loss_func,None

    def __enter__(self):
        if hasattr(self.loss_func, 'reduction'):
            self.old_red = getattr(self.loss_func, 'reduction')
            setattr(self.loss_func, 'reduction', 'none')
            return self.loss_func
        else: return partial(self.loss_func, reduction='none')

    def __exit__(self, type, value, traceback):
        if self.old_red is not None: setattr(self.loss_func, 'reduction', self.old_red)

class MixUp(Callback):
    _order = 90 #Runs after normalization and cuda
    def __init__(self, alpha:float=0.4): self.distrib = Beta(tensor([alpha]), tensor([alpha]))

    def begin_fit(self): self.old_loss_func,self.run.loss_func = self.run.loss_func,self.loss_func

    def begin_batch(self):
        if not self.in_train: return #Only mixup things during training
        lambd = self.distrib.sample((self.yb.size(0),)).squeeze().to(self.xb.device)
        self.lambd = torch.cat([lambd[:,None], 1-lambd[:,None]], 1).max(1)[0]
        shuffle = torch.randperm(self.yb.size(0)).to(self.xb.device)
        xb1,self.yb1 = self.xb[shuffle],self.yb[shuffle]
        self.run.xb = self.xb * self.lambd[:,None,None,None] + xb1 * (1-self.lambd)[:,None,None,None]

    def after_fit(self): self.run.loss_func = self.old_loss_func

    def loss_func(self, pred, yb):
        if not self.in_train: return self.old_loss_func(pred, yb)
        with NoneReduce(self.old_loss_func) as loss_func:
            loss1 = loss_func(pred, yb)
            loss2 = loss_func(pred, self.yb1)
        return (loss1 * self.lambd + loss2 * (1-self.lambd)).mean()