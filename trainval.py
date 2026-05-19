import torch
import pandas as pd
import os
import time
import numpy as np
import argparse
from src import models
from src import datasets
from src import utils as ut
from torch.utils.data import DataLoader


def trainval(exp_dict, savedir, args):
    # Set seed and device
    # ===================
    seed = 42 + exp_dict['runs']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        device = 'cuda'
        torch.cuda.manual_seed_all(seed)
        assert torch.cuda.is_available(), 'cuda is not, available please run with "-c 0"'
    else:
        device = 'cpu'

    print('Running on device: %s' % device)

    # Load Datasets
    # ==================
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     split='train',
                                     datadir=args.datadir,
                                     exp_dict=exp_dict)

    train_loader = DataLoader(train_set,
                              drop_last=True,
                              shuffle=True,
                              sampler=None,
                              batch_size=exp_dict["batch_size"])

    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                   split='val',
                                   datadir=args.datadir,
                                   exp_dict=exp_dict)

    # Load Model
    # ==================
    model = models.get_model(train_loader, exp_dict, device=device)
    model_path = os.path.join(savedir, "model.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    if os.path.exists(score_list_path):
        # resume experiment
        score_list = ut.load_pkl(score_list_path)
        model.set_state_dict(torch.load(model_path))
        s_epoch = score_list[-1]["epoch"] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Train and Val
    # ==============
    for epoch in range(s_epoch, exp_dict["max_epoch"]):
        # Set seed
        seed = epoch + exp_dict.get('runs', 0)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if epoch == 0:
            # Validate one epoch
            train_dict = model.val_on_dataset(train_set, metric=exp_dict["loss_func"])
            val_dict = model.val_on_dataset(val_set, metric=exp_dict["acc_func"])

        else:
            # Train one epoch
            model.train_on_loader(train_loader)

            # Validate one epoch
            train_dict = model.val_on_dataset(train_set, metric=exp_dict["loss_func"])
            val_dict = model.val_on_dataset(val_set, metric=exp_dict["acc_func"])

        # Record metrics
        score_dict = {"epoch": epoch}
        score_dict["train_" + train_dict["metric"]] = train_dict["score"]
        score_dict["val_" + val_dict["metric"]] = val_dict["score"]
        score_dict["step_size"] = model.opt.state.get("step_size", {})
        score_dict["step_size_avg"] = model.opt.state.get("step_size_avg", {})
        score_dict["n_forwards"] = model.opt.state.get("n_forwards", {})
        score_dict["n_backwards"] = model.opt.state.get("n_backwards", {})
        score_dict["grad_norm"] = model.opt.state.get("grad_norm", {})
        score_dict.update(model.opt.state["gv_stats"])

        # Add score_dict to score_list
        score_list += [score_dict]

        # Report and save
        df = pd.DataFrame(score_list)
        print(df.tail())
        ut.save_pkl(score_list_path, score_list)
        ut.save_json(os.path.join(savedir, "score_list.json"), score_list)
        ut.torch_save(model_path, model.get_state_dict())
        print("Saved: %s" % savedir)

        df.to_csv(os.path.join(savedir, "score_list.csv"), index=False)

    print("Done with Experiment!")

if __name__ == "__main__":
    import exp_configs as exp_configs
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--exp_group", default="mnist")
    parser.add_argument('-sb', '--savedir_base', default="results")
    parser.add_argument('-d', '--datadir', default="data")
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-c", "--cuda", default=0, type=int)
    parser.add_argument("-v", "--visualize", default=1, type=int)
    args, others = parser.parse_known_args()

    exp_group = exp_configs.EXP_GROUPS[args.exp_group]
    for exp_dict in exp_group:
        exp_hash = ut.get_exp_hash(exp_dict)
        savedir = os.path.join(args.savedir_base, exp_hash)

        # Reset experiment if already exists
        if args.reset:
            if os.path.exists(savedir):
                print("Experiment already exists, deleting and resetting: %s" % savedir)
                import shutil
                shutil.rmtree(savedir)

        # Run experiment
        if os.path.exists(savedir):
            print("Experiment already exists, skipping: %s" % savedir)
            continue
        else:
            os.makedirs(savedir, exist_ok=True)
            ut.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)
            print("Running experiment and saving to: %s" % savedir)
            trainval(exp_dict, savedir, args)

    # visualize experiments
    if args.visualize:
        ut.visualize(exp_group, args.savedir_base,
                     name=args.exp_group,
                     x_col="epoch",
                     y_cols=("train_" + exp_dict["loss_func"], "val_" + exp_dict["acc_func"]))
