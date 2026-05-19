import pickle
import json
import os
import itertools
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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

    

def flatten_dict(d, parent_key=""):
    items = []
    for k, v in d.items():
        full_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, full_key).items())
        else:
            items.append((full_key, v))
    return dict(items)


def _get_hparams(exp_group):
    flat_group = [flatten_dict(e) for e in exp_group]
    all_keys = set().union(*[e.keys() for e in flat_group])
    return [k for k in all_keys if len({str(e.get(k)) for e in flat_group}) > 1]


def visualize(exp_group, savedir_base, name="results", x_col="epoch", y_cols=("train_loss", "val_score")):
    plots_dir = os.path.join(savedir_base, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    hparam_keys = _get_hparams(exp_group)
    fig, axes = plt.subplots(1, len(y_cols), figsize=(6 * len(y_cols), 4))
    if len(y_cols) == 1:
        axes = [axes]

    for exp_dict in exp_group:
        exp_hash = get_exp_hash(exp_dict)
        score_list_path = os.path.join(savedir_base, exp_hash, "score_list.pkl")
        if not os.path.exists(score_list_path):
            continue

        df = pd.DataFrame(load_pkl(score_list_path))
        flat = flatten_dict(exp_dict)
        label = "_".join(f"{k}={flat.get(k)}" for k in hparam_keys)

        for ax, col in zip(axes, y_cols):
            if col in df.columns:
                ax.plot(df[x_col], df[col], label=label)
                ax.set_xlabel(x_col)
                ax.set_ylabel(col)
                ax.set_title(col)

    for ax in axes:
        ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)
    print("Plots saved to: %s" % plots_dir)


def cartesian_exp_group(exp_config):
    keys, values = zip(*exp_config.items())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def get_exp_hash(exp_dict):
    import hashlib
    flat = flatten_dict(exp_dict)
    name = "_".join(f"{k}_{v}" for k, v in sorted(flat.items()))
    suffix = "_" + hashlib.md5(name.encode()).hexdigest()[:8]
    max_len = 255 - len(suffix.encode())
    return name.encode()[:max_len].decode(errors="ignore") + suffix


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


    





