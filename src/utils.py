import os

from src import models
from haven import haven_utils as hu


def compute_fstar(train_set, loss_function, savedir_base, exp_dict):
    fstar_dict = {'dataset':exp_dict['dataset'], 
                      'model':exp_dict['model'], 
                      'loss_func':exp_dict['loss_func']}
    fstar_id = hu.hash_dict(fstar_dict)
    model_func = lambda: models.get_model(exp_dict["model"], train_set=train_set).cuda()
    train_set.compute_fstar(model_func, loss_function, 
                            fname=os.path.join(savedir_base,
                                                '%s_fstar.pkl' % (fstar_id)))  