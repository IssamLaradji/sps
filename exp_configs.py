from haven import haven_utils as hu
import itertools 


# datasets
kernel_datasets = ["mushrooms",
                #    "w8a", "ijcnn", 
                #    "rcv1"
                   ]

# define runs
run_list = [0]

# define optimizers
c_list = [0.2]
sps_list = []

for c, adapt_flag in itertools.product(c_list, ['smooth_iter']):
    sps_list += [{'name':"sps", "c":c, 'adapt_flag':adapt_flag}]

opt_list =  sps_list + [{'name': 'adam'}]

EXP_GROUPS = {}

# define interpolation exp groups
EXP_GROUPS['kernel'] = hu.cartesian_exp_group({"dataset":kernel_datasets,
                                "model":["linear"],
                                "loss_func": ['logistic_loss'],
                                "acc_func": ["logistic_accuracy"],
                                "opt": opt_list ,
                                "batch_size":[100],
                                "max_epoch":[35],
                                "runs":run_list})

EXP_GROUPS['mf'] = hu.cartesian_exp_group({"dataset":["matrix_fac"],
                                "model":["matrix_fac_1", "matrix_fac_4", "matrix_fac_10", "linear_fac"],
                                "loss_func": ["squared_loss"],
                                "opt": opt_list,
                                "acc_func":["mse"],
                                "batch_size":[100],
                                "max_epoch":[50],
                                "runs":run_list})

EXP_GROUPS['mnist'] = hu.cartesian_exp_group({"dataset":["mnist"],
                                "model":["mlp"],
                                "loss_func": ["softmax_loss"],
                                "opt":[{'name':"sps", "c":c, 
                                        'adapt_flag':'smooth_iter',
                                         'centralize_grad':True}] +  opt_list,
                                "acc_func":["softmax_accuracy"],
                                "batch_size":[128],
                                "max_epoch":[200],
                                "runs":run_list})

EXP_GROUPS['deep'] = (hu.cartesian_exp_group({"dataset":["cifar10"],
                                "model":["resnet34", "densenet121"],
                                "loss_func": ["softmax_loss"],
                                "opt": opt_list,
                                "acc_func":["softmax_accuracy"],
                                "batch_size":[128],
                                "max_epoch":[200],
                                "runs":run_list}) +

                           hu.cartesian_exp_group({"dataset":["cifar100"],
                                        "model":["resnet34_100", "densenet121_100"],
                                        "loss_func": ["softmax_loss"],
                                        "opt": opt_list,
                                        "acc_func":["softmax_accuracy"],
                                        "batch_size":[128],
                                        "max_epoch":[200],
                                        "runs":run_list})
                            )

EXP_GROUPS['cifar'] = hu.cartesian_exp_group({"dataset":["cifar10"],
                                "model":["resnet34"],
                                "loss_func": ["softmax_loss"],
                                "opt":  opt_list + [{'name':"sps", "c":c, 
                                                   'adapt_flag':'smooth_iter',
                                                   'centralize_grad':True}] ,
                                "acc_func":["softmax_accuracy"],
                                "batch_size":[128],
                                "max_epoch":[200],
                                "runs":[0]})
                            

# define non-interpolation exp groups
eta_max_list = [1, 5, 100]
c_list = [0.5]

sps_l2_list = []
for c, eta_max in itertools.product(c_list, eta_max_list):
        sps_l2_list += [{'name':"sps",  "c":c, 
                             'fstar_flag':True, 'eps':0,
                             'adapt_flag':'constant', 
                             'eta_max':eta_max}]
sps_list = []
for c, eta_max in itertools.product(c_list, eta_max_list):
        sps_list += [{'name':"sps",  "c":c, 
                             'fstar_flag':False, 'eps':0,
                             'adapt_flag':'constant', 
                             'eta_max':eta_max}]

sgd_list = [{'name':"sgd", 
        "lr":10.0},{'name':"sgd", 
        "lr":1.0}, {'name':"sgd", 
        "lr":1e-3}, {'name':"sgd", 
        "lr":1e-1}, {'name':"sgd", 
        "lr":1e-2}] 

EXP_GROUPS['syn_l2'] = (hu.cartesian_exp_group({"dataset":['syn'],
                "model":["logistic"],
                "loss_func": [
                             'logistic_l2_loss', 
                             ],
                "acc_func": ["logistic_accuracy"],
                "opt": sps_l2_list + sgd_list,   
                "batch_size":[1],
                "max_epoch":[50],
                "runs":run_list}))

EXP_GROUPS['syn'] = (hu.cartesian_exp_group({"dataset":['syn'],
                "model":["logistic"],
                "loss_func": [
                             'logistic_loss', 
                             ],
                "acc_func": ["logistic_accuracy"],
                "opt": sps_list + sgd_list,   
                "batch_size":[1],
                "max_epoch":[50],
                "runs":run_list}))

