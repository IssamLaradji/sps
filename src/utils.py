import hashlib 
import pickle
import json
import os
import itertools
import torch
import numpy as np
import tqdm


def opt_step(name, opt, model, batch, loss_function, use_backpack, epoch):
    device = next(model.parameters()).device
    images, labels = batch["images"].to(device=device), batch["labels"].to(device=device)

    if (name in ['adaptive_second']):
        closure = lambda for_backtracking=False : loss_function(model, images, labels, backwards=False, 
                                                                backpack=(use_backpack and not for_backtracking))
        loss = opt.step(closure)
                
    elif (name in ["sgd_armijo", "ssn", 'adaptive_first', 'l4', 'ali_g']):
        closure = lambda : loss_function(model, images, labels, backwards=False, backpack=use_backpack)
        loss = opt.step(closure)
                
    elif (name in ['sps']):
        closure = lambda : loss_function(model, images, labels, backwards=False, backpack=use_backpack)
        loss = opt.step(closure, batch)

    elif (name in ["adam", "adagrad", 'radam', 'plain_radam', 'adabound']):
        loss = loss_function(model, images, labels, backpack=use_backpack)
        loss.backward()
        opt.step()

    else:
        raise ValueError('%s optimizer does not exist' % name)
    
    return loss

    

def save_pkl(fname, data):
    """Save data in pkl format."""
    # Save file
    fname_tmp = fname + "_tmp.pkl"
    with open(fname_tmp, "wb") as f:
        pickle.dump(data, f)
    os.rename(fname_tmp, fname)


def load_pkl(fname):
    """Load the content of a pkl file."""
    with open(fname, "rb") as f:
        return pickle.load(f)

def load_json(fname, decode=None):
    with open(fname, "r") as json_file:
        d = json.load(json_file)

    return d

def save_json(fname, data):
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)

def torch_save(fname, obj):
    """"Save data in torch format."""
    # Define names of temporal files
    fname_tmp = fname + ".tmp"

    torch.save(obj, fname_tmp)
    os.rename(fname_tmp, fname)

def read_text(fname):
    # READS LINES
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # lines = [line.decode('utf-8').strip() for line in f.readlines()]
    return lines


def compute_fstar(model, train_set):
    from src.optimizers import sls 
    model.zero_grad()
    for i in range(len(model.params)):
        model.params[i].data[:] = 0
    # loss = closure()
    opt = sls.Sls(model.params, n_batches_per_epoch=1.0, c=0.5)
    for i in range(500):
        opt.zero_grad()
        loss = opt.step(closure).item()
        
        grad_current = ut.get_grad_list(model.params)
        grad_norm = ut.compute_grad_norm(grad_current)
        if np.isnan(loss):
            print('nan')
        # print(i, loss)
        if grad_norm < 1e-8:
            break
    return loss


def update_trainloader_and_opt(train_set, opt, batch_size, n_train, batch_grow_factor, batch_size_max, 
                               ssn_flag=True):
    n_iters = (n_train // batch_size)

    batch_size_new = batch_size * batch_grow_factor ** n_iters
    batch_size_new = min(int(batch_size_new), batch_size_max)
    batch_size_new = min(batch_size_new, n_train)
    train_loader = torch.utils.data.DataLoader(train_set, 
                                         batch_size=batch_size_new,
                                          drop_last=False, 
                                         shuffle=True)
                                         
    opt.n_batches_in_epoch = (n_train // batch_size_new)

    
    if ssn_flag:
        from ssn import newton
        assert isinstance(opt, newton.SSNArmijo)
        opt.lm = opt.lm / (np.sqrt(batch_grow_factor ** n_iters))

        print('LM regularization = ', opt.lm)

    return train_loader, opt
    





