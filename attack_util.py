import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


### Do not modif the following codes
class ctx_noparamgrad(object):
    def __init__(self, module):
        self.prev_grad_state = get_param_grad_state(module)
        self.module = module
        set_param_grad_off(module)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        set_param_grad_state(self.module, self.prev_grad_state)
        return False


def get_param_grad_state(module):
    return {param: param.requires_grad for param in module.parameters()}


def set_param_grad_off(module):
    for param in module.parameters():
        param.requires_grad = False


def set_param_grad_state(module, grad_state):
    for param in module.parameters():
        param.requires_grad = grad_state[param]


### Ends


### PGD Attack
class PGDAttack():
    def __init__(self, attack_step=10, eps=8 / 255, alpha=2 / 255, loss_type='ce', targeted=True,
                 num_classes=10, device="cpu"):
        '''
        attack_step: number of PGD iterations
        eps: attack budget
        alpha: PGD attack step size
        '''
        ### Your code here
        self.attack_step = attack_step
        self.eps = eps
        self.alpha = alpha
        self.lower_limit = 0
        self.upper_limit = 1
        self.targeted = targeted
        self.num_classes = num_classes
        self.device = device
        if loss_type == 'ce':
            self.loss = self.ce_loss
        elif loss_type == 'cw':
            self.loss = self.cw_loss
        else:
            print(f"Attack loss {loss_type} not support!")
            exit()
        ### Your code ends

    def ce_loss(self, logits, y):
        ### Your code here
        out = logits.clone()
        gt = y.clone()

        # turn logits into probabilities
        out_exp = torch.exp(out)
        sum_exp = torch.sum(out_exp, dim=1).unsqueeze(1)
        pred = out_exp / sum_exp

        # eliminate the potential 0s in the predicted probabilities
        very_small = 1e-12
        pred = torch.clamp(pred, very_small, 1.-very_small)

        N = gt.shape[0]
        # only the predicted values with ground true indexes need to be log
        batch_loss = -torch.log(pred[range(N), gt.long()])
        loss = torch.mean(batch_loss)
        return loss
        ### Your code ends

    def cw_loss(self, logits, y):
        ### Your code here
        out = logits.clone()
        gt = y.clone()

        pass
        ### Your code ends

    def perturb(self, model: nn.Module, X, y):
        X.requires_grad = False
        delta = torch.zeros_like(X, requires_grad=True)
        ### Your code here
        model.eval()
        for i in range(self.attack_step):
            output = model(X+delta)
            loss = self.loss(output, y)
            model.zero_grad()
            loss.backward()

            grad = delta.grad.detach()
            sign = grad.sign()
            delta.data = delta.data - self.alpha * sign  # descent update
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)  # projection
            delta.data = torch.clamp(delta.data, self.lower_limit-X.data, self.upper_limit-X.data)
            delta.grad.zero_()
            print(f"PGD step {i} done!")
        ### Your code ends

        return delta


### FGSMAttack
'''
Technically you can transform your PGDAttack to FGSM Attack by controling parameters like `attack_step`. 
If you do that, you do not need to implement FGSM in this class.
'''


class FGSMAttack():
    def __init__(self, eps=8 / 255, loss_type='ce', targeted=True, num_classes=10, device="cpu"):
        self.eps = eps
        self.device = device
        self.num_classes = num_classes
        self.targeted = targeted
        if loss_type == 'ce':
            self.loss = self.ce_loss
        else:
            print(f"Attack loss {loss_type} not support!")
            exit()

    def perturb(self, model: nn.Module, X, y):
        X.requires_grad = True
        delta = torch.ones_like(X, requires_grad=False)
        ### Your code here

        # set eval mode to disable dropout and batch normalization layers and ensures that the model's
        # parameters are not updated during the forward pass
        model.eval()
        output = model(X)
        loss = self.loss(output, y)
        # clear the gradients from any previous backward pass before computing the gradients for the current attack
        model.zero_grad()
        # calculate the gradient w.r.t X
        loss.backward()

        # fgsm: delta=-eps*sign(grad(loss)|X)
        grad = X.grad.detach()
        sign = grad.sign()
        delta = -1 * self.eps * sign
        X.grad.zero_()

        ### Your code ends
        return delta

    def ce_loss(self, logits, y):
        ### Your code here
        out = logits.clone()
        gt = y.clone()

        # turn logits into probabilities
        out_exp = torch.exp(out)
        sum_exp = torch.sum(out_exp, dim=1).unsqueeze(1)
        pred = out_exp / sum_exp

        # eliminate the potential 0s in the predicted probabilities
        very_small = 1e-12
        pred = torch.clamp(pred, very_small, 1.-very_small)

        N = gt.shape[0]
        # only the predicted values with ground true indexes need to be log
        batch_loss = -torch.log(pred[range(N), gt.long()])
        loss = torch.mean(batch_loss)
        return loss
        ### Your code ends
