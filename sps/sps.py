import numpy as np
import torch
import time
import copy


class Sps(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=0.5,
                 gamma=2.0,
                 eta_max=None,
                 adapt_flag='smooth_iter',
                 fstar_flag=None,
                 eps=1e-8,
                 centralize_grad_norm=False,
                 centralize_grad=False):
        params = list(params)
        super().__init__(params, {})
        self.eps = eps
        self.params = params
        self.c = c
        self.centralize_grad_norm = centralize_grad_norm
        self.centralize_grad = centralize_grad

        if centralize_grad:
            assert self.centralize_grad_norm is False

        self.eta_max = eta_max
        self.gamma = gamma
        self.init_step_size = init_step_size
        self.adapt_flag = adapt_flag
        self.state['step'] = 0

        self.state['step_size'] = init_step_size
        self.step_size_max = 0.
        self.n_batches_per_epoch = n_batches_per_epoch

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        self.fstar_flag = fstar_flag

    def step(self, closure=None, loss=None, batch=None):
        if loss is None and closure is None:
            raise ValueError('please specify either closure or loss')

        if loss is not None:
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss)

        # increment step
        self.state['step'] += 1

        # get fstar
        if self.fstar_flag:
            fstar = float(batch['meta']['fstar'].mean())
        else:
            fstar = 0.

        # get loss and compute gradients
        if loss is None:
            loss = closure()
        else:
            assert closure is None, 'if loss is provided then closure should beNone'

        # save the current parameters:
        grad_current = get_grad_list(self.params, centralize_grad=self.centralize_grad)
        grad_norm = compute_grad_norm(grad_current, centralize_grad_norm=self.centralize_grad_norm)

        if grad_norm < 1e-8:
            step_size = 0.
        else:
            # adapt the step size
            if self.adapt_flag in ['constant']:
                # adjust the step size based on an upper bound and fstar
                step_size = (loss - fstar) / \
                    (self.c * (grad_norm)**2 + self.eps)
                if loss < fstar:
                    step_size = 0.
                else:
                    if self.eta_max is None:
                        step_size = step_size.item()
                    else:
                        step_size = min(self.eta_max, step_size.item())

            elif self.adapt_flag in ['smooth_iter']:
                # smoothly adjust the step size
                step_size = loss / (self.c * (grad_norm)**2 + self.eps)
                coeff = self.gamma**(1./self.n_batches_per_epoch)
                step_size = min(coeff * self.state['step_size'],
                                step_size.item())
            else:
                raise ValueError('adapt_flag: %s not supported' %
                                 self.adapt_flag)

            # update with step size
            sgd_update(self.params, step_size, grad_current)

        # update state with metrics
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1
        self.state['step_size'] = step_size
        self.state['grad_norm'] = grad_norm.item()

        if torch.isnan(self.params[0]).sum() > 0:
            raise ValueError('Got NaNs')

        return float(loss)

# utils
# ------------------------------
def compute_grad_norm(grad_list, centralize_grad_norm=False):
    grad_norm = 0.
    for g in grad_list:
        if g is None:
            continue

        if g.dim() > 1 and centralize_grad_norm: 
            # centralize grads 
            g.add_(-g.mean(dim = tuple(range(1,g.dim())), keepdim = True))

        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm


def get_grad_list(params, centralize_grad=False):
    grad_list = []
    for p in params:
        g = p.grad
        if g is None:
            g = 0.
        else:
            g = p.grad.data
            if len(list(g.size()))>1 and centralize_grad:
                # centralize grads
                g.add_(-g.mean(dim = tuple(range(1,len(list(g.size())))), 
                       keepdim = True))
                   
        grad_list += [g]        
                   
    return grad_list


def sgd_update(params, step_size, grad_current):
    for p, g in zip(params, grad_current):
        p.data.add_(- step_size, g)
