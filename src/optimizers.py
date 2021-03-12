import numpy as np
import torch
import time
import copy
from sps import Sps


def get_optimizer(opt_dict, params, train_loader, exp_dict):
    """
    opt: name or dict
    params: model parameters
    n_batches_per_epoch: b/n
    """
    opt_name = opt_dict['name']
    # our optimizers   
    n_train = len(train_loader.dataset)
    batch_size = train_loader.batch_size
    n_batches_per_epoch = n_train / float(batch_size) 
    if opt_name == 'sps':
        opt = Sps(params, c=opt_dict["c"], 
                        n_batches_per_epoch=n_batches_per_epoch, 
                        adapt_flag=opt_dict.get('adapt_flag'),
                        fstar_flag=opt_dict.get('fstar_flag'),
                        eta_max=opt_dict.get('eta_max'),
                        eps=opt_dict.get('eps', 0),
                        centralize_grad=opt_dict.get("centralize_grad", False),
                        centralize_grad_norm=opt_dict.get("centralize_grad_norm", False), )
    # others        
    elif opt_name == "adam":
        opt = torch.optim.Adam(params, lr=opt_dict.get('lr', 1e-3))

    elif opt_name == "adagrad":
        opt = torch.optim.Adagrad(params, lr=opt_dict.get('lr', 0.01))

    elif opt_name == 'sgd':
        opt = torch.optim.SGD(params, lr=opt['lr'])

    elif opt_name == "sgd-m":
        opt = torch.optim.SGD(params, lr=opt_dict.get('lr', 1e-3), momentum=0.9)

    elif opt_name == 'rms':
        opt = torch.optim.RMSprop(params)

    else:
        raise ValueError("opt %s does not exist..." % opt_name)

    return opt

