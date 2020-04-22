import time
import copy
import numpy as np
import torch


class Sps(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=0.5,
                 gamma=2.0,
                 eta_max=10,
                 adapt_flag='smooth_iter',
                 fstar_flag=None,
                 eps=1e-8):
        defaults = {k: v for k,v in locals().items() if k not in ('params',)}
        super().__init__(params, defaults)

    def step(self, closure=None, loss=None, batch=None):
        if loss is None and closure is None:
            raise ValueError('please specify either closure or loss')

        # get loss and compute gradients
        if loss is None:
            loss = closure()
        else:
            assert closure is None, 'if loss is provided then closure should be None'
            loss = torch.as_tensor(loss)

        for group in self.param_groups:
            # more initializations
            if 'step' not in group:
                group['step'] = 0
                group['step_size_avg'] = 0.
                group['step_size'] = group['init_step_size']
                group['n_forwards'] = 0
                group['n_backwards'] = 0

            # increment step
            group['step'] += 1

            # get fstar
            if group['fstar_flag']:
                fstar = float(batch['meta']['fstar'].mean())
            else:
                fstar = 0.

            # save the current parameters:
            params_current = copy.deepcopy(group['params'])
            grad_current = get_grad_list(group['params'])
            grad_norm = compute_grad_norm(grad_current)

            if grad_norm < 1e-8:
                step_size = 0.
            else:
                # adapt the step size
                if group['adapt_flag'] in ['constant']:
                    # adjust the step size based on an upper bound and fstar
                    step_size = (loss - fstar) / (group['c'] * (grad_norm)**2 + group['eps'])
                    if loss < fstar:
                        step_size = 0.
                    else:
                        if group['eta_max'] is None:
                            step_size = step_size.item()
                        else:
                            step_size = min(group['eta_max'], step_size.item())

                elif group['adapt_flag'] in ['smooth_iter']:
                    # smoothly adjust the step size
                    step_size = loss / (group['c'] * (grad_norm)**2 + group['eps'])
                    coeff = group['gamma']**(1./group['n_batches_per_epoch'])
                    step_size = min(coeff * group['step_size'], step_size.item())
                else:
                    raise ValueError('adapt_flag: %s not supported' % group['adapt_flag'])

                # update with step size
                sgd_update(group['params'], step_size,params_current, grad_current)

            # update state with metrics
            if group['step'] % int(group['n_batches_per_epoch']) == 1:
                # reset step size avg for each new epoch
                group['step_size_avg'] = 0.

            group['step_size_avg'] += (step_size / group['n_batches_per_epoch'])

            group['n_forwards'] += 1
            group['n_backwards'] += 1
            group['step_size'] = step_size
            group['grad_norm'] = grad_norm.item()

            if torch.isnan(group['params'][0]).sum() > 0:
                raise ValueError('Got NaNs')

        return loss

# utils
# ------------------------------


def compute_grad_norm(grad_list):
    grad_norm = 0.
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm


def get_grad_list(params):
    return [p.grad for p in params]


def sgd_update(params, step_size, params_current, grad_current):
    for p_next, p_current, g_current in zip(params, params_current, grad_current):
        if g_current is not None:
            p_next.data = p_current - step_size * g_current
