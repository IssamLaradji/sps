# -*- coding: utf-8 -*-

import os, pprint, tqdm
import numpy as np
import pandas as pd
from haven import haven_utils as hu 
from haven import haven_img as hi
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from . import base_classifiers
from . import optimizers


def get_model(train_loader, exp_dict, device):
    return Classifier(train_loader, exp_dict, device)

    
class Classifier(torch.nn.Module):
    def __init__(self, train_loader, exp_dict, device):
        super().__init__()
        self.exp_dict = exp_dict
        self.device = device
        
        self.model_base = base_classifiers.get_classifier(exp_dict['model'], train_set=train_loader.dataset)

        # Load Optimizer
        self.to(device=self.device)
        self.opt = optimizers.get_optimizer(opt_dict=exp_dict["opt"],
                                       params=self.parameters(),
                                       train_loader=train_loader,                                
                                       exp_dict=exp_dict)
        
    def train_on_loader(self, train_loader):
        self.train()

        pbar = tqdm.tqdm(train_loader)
        for batch in pbar:
            score_dict = self.train_on_batch(batch)
            pbar.set_description(f'Training - {score_dict["train_loss"]:.3f}')
            
        return score_dict

    def get_state_dict(self):
        state_dict = {"model": self.model_base.state_dict(),
                      "opt": self.opt.state_dict()}

        return state_dict

    def set_state_dict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"])
        self.opt.load_state_dict(state_dict["opt"])
    
    def val_on_dataset(self, dataset, metric, name):
        self.eval()

        metric_function = get_metric_function(metric)
        loader = torch.utils.data.DataLoader(dataset, drop_last=False, batch_size=self.exp_dict['batch_size'])

        score_sum = 0.
        pbar = tqdm.tqdm(loader)
        for batch in pbar:
            images, labels = batch["images"].to(device=self.device), batch["labels"].to(device=self.device)
            score_sum += metric_function(self.model_base, images, labels).item() * images.shape[0]    
            score = float(score_sum / len(loader.dataset))
            
            pbar.set_description(f'Validating {metric}: {score:.3f}')

        return {f'{dataset.split}_{name}': score}


    def val_on_loader(self, batch):
        pass 
    
    def train_on_batch(self, batch):
        self.opt.zero_grad()
        loss_function = get_metric_function(self.exp_dict['loss_func'])
        images, labels = batch["images"].to(device=self.device), batch["labels"].to(device=self.device)
        use_backpack = False

        name = self.exp_dict['opt']['name']
        if (name in ['adaptive_second']):
            closure = lambda for_backtracking=False : loss_function(self.model_base, images, labels, backwards=False, 
                                                                    backpack=(use_backpack and not for_backtracking))
            loss = self.opt.step(closure)
                    
        elif (name in ["sgd_armijo", "ssn", 'adaptive_first', 'l4', 'ali_g', 'sgd_goldstein', 'sgd_nesterov', 'sgd_polyak', 'seg']):
            closure = lambda : loss_function(self.model_base, images, labels, backwards=False, backpack=use_backpack)
            loss = self.opt.step(closure)

        else:
            closure = lambda : loss_function(self.model_base, images, labels, backwards=True, backpack=use_backpack)
            loss = self.opt.step(closure=closure)
            
        return {'train_loss': float(loss)}


        

# Metrics
def get_metric_function(metric):
    if metric == "logistic_accuracy":
        return logistic_accuracy

    if metric == "softmax_accuracy":
        return softmax_accuracy

    elif metric == "softmax_loss":
        return softmax_loss

    elif metric == "logistic_l2_loss":
        return logistic_l2_loss
    
    elif metric == "squared_hinge_l2_loss":
        return squared_hinge_l2_loss

    elif metric == "squared_l2_loss":
        return squared_l2_loss

    elif metric == 'perplexity':
        return lambda *args, **kwargs: torch.exp(softmax_loss(*args, **kwargs)) 
 
    elif metric == "logistic_loss":
        return logistic_loss

    elif metric == "squared_hinge_loss":
        return squared_hinge_loss

    elif metric == "mse":
        return mse_score

    elif metric == "squared_loss":
        return squared_loss



def logistic_l2_loss(model, images, labels, backwards=False, reduction="mean", backpack=False):
    logits = model(images)
    criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)
    loss = criterion(logits.view(-1), labels.view(-1))

    w = 0.
    for p in model.parameters():
        w += (p**2).sum()

    loss += 1e-4 * w

    if backwards and loss.requires_grad:
        loss.backward()

    return loss
    
def softmax_loss(model, images, labels, backwards=False, reduction="mean", backpack=False):
    logits = model(images)
    criterion = torch.nn.CrossEntropyLoss(reduction=reduction)
    if backpack:
        criterion = extend(criterion)
    loss = criterion(logits, labels.long().view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def logistic_loss(model, images, labels, backwards=False, reduction="mean", backpack=False):
    logits = model(images)
    criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)
    if backpack:
        criterion = extend(criterion)
    loss = criterion(logits.view(-1), labels.float().view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def squared_loss(model, images, labels, backwards=False, reduction="mean", backpack=False):
    logits = model(images)
    criterion = torch.nn.MSELoss(reduction=reduction)
    if backpack:
        criterion = extend(criterion)
    loss = criterion(logits.view(-1), labels.view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def mse_score(model, images, labels):
    logits = model(images).view(-1)
    mse = ((logits - labels.view(-1))**2).float().mean()

    return mse

def squared_hinge_loss(model, images, labels, backwards=False, reduction="mean", backpack=False):
    margin=1.
    logits = model(images).view(-1)

    y = 2*labels - 1

    loss = (torch.max( torch.zeros_like(y) , 
                torch.ones_like(y) - torch.mul(y, logits)))**2

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    
    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def add_l2(model):
    w = 0.
    for p in model.parameters():
        w += (p**2).sum()

    loss = 1e-4 * w

    return loss

def squared_hinge_l2_loss(model, images, labels, backwards=False, reduction="mean", backpack=False):
    loss = squared_hinge_loss(model, images, labels, backwards=False, reduction=reduction)
    loss += add_l2(model)

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def squared_l2_loss(model, images, labels, backwards=False, reduction="mean", backpack=False):
    loss = squared_loss(model, images, labels, backwards=False, reduction=reduction)
    loss += add_l2(model)
    
    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def logistic_accuracy(model, images, labels):
    logits = torch.sigmoid(model(images)).view(-1)
    pred_labels = (logits > 0.5).float().view(-1)
    acc = (pred_labels == labels).float().mean()

    return acc

def softmax_accuracy(model, images, labels):
    logits = model(images)
    pred_labels = logits.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()

    return acc


