import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from sklearn import metrics
from haven import haven_utils as hu
from torch.utils.data import Dataset
import tqdm
from . import optimizers
import os
import urllib

import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from torchvision.datasets import MNIST


LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
LIBSVM_DOWNLOAD_FN = {"rcv1"       : "rcv1_train.binary.bz2",
                      "mushrooms"  : "mushrooms",
                      "ijcnn"      : "ijcnn1.tr.bz2",
                      "w8a"        : "w8a"}


def get_dataset(dataset_name, train_flag, datadir, exp_dict):
    if dataset_name == "mnist":
        dataset = torchvision.datasets.MNIST(datadir, train=train_flag,
                               download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.5,), (0.5,))
                               ]))

    if dataset_name == "cifar10":
        if train_flag:
            transform_function = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_function = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        dataset = torchvision.datasets.CIFAR10(
            root=datadir,
            train=train_flag,
            download=True,
            transform=transform_function)

    if dataset_name == "cifar100":
        if train_flag:
            transform_function = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_function = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        dataset = torchvision.datasets.CIFAR100(
            root=datadir,
            train=train_flag,
            download=True,
            transform=transform_function)

    if dataset_name in ['syn']:
        bias = 1; 
        scaling = 10; 
        sparsity = 10; 
        solutionSparsity = 0.1;

        n = 1000
        p = 100
            
        A = np.random.randn(n,p)+bias;
        A = A.dot(np.diag(scaling* np.random.randn(p)))
        A = A * (np.random.rand(n,p) < (sparsity*np.log(n)/n));
        w = np.random.randn(p) * (np.random.rand(p) < solutionSparsity);

        b = np.sign(A.dot(w));
        b = b * np.sign(np.random.rand(n)-0.1);
        labels = np.unique(b)
        A = A / np.linalg.norm(A, axis=1)[:, None].clip(min=1e-6)
        A = A * 2
        b[b==labels[0]] = 0
        b[b==labels[1]] = 1

        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(A), torch.FloatTensor(b))

        return DatasetWrapper(dataset)

        
    if dataset_name in ['mushrooms', 'w8a', 'rcv1', 'ijcnn']:
        sigma_dict = {"mushrooms": 0.5,
                      "w8a":20.0,
                      "rcv1":0.25 ,
                      "ijcnn":0.05}

        X, y = load_libsvm(dataset_name, data_dir=datadir)

        labels = np.unique(y)

        y[y==labels[0]] = 0
        y[y==labels[1]] = 1
        # splits used in experiments
        splits = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=9513451)
        X_train, X_test, Y_train, Y_test = splits

        if train_flag:
            # fname_rbf = "%s/rbf_%s_%s_train.pkl" % (datadir, dataset_name, sigma_dict[dataset_name])
            fname_rbf = "%s/rbf_%s_%s_train.npy" % (datadir, dataset_name, sigma_dict[dataset_name])
            if os.path.exists(fname_rbf):
                k_train_X = np.load(fname_rbf)
            else:
                k_train_X = rbf_kernel(X_train, X_train, sigma_dict[dataset_name])
                np.save(fname_rbf, k_train_X)
                print('%s saved' % fname_rbf)

            X_train = k_train_X
            X_train = torch.FloatTensor(X_train)
            Y_train = torch.FloatTensor(Y_train)

            dataset = torch.utils.data.TensorDataset(X_train, Y_train)

        else:
            fname_rbf = "%s/rbf_%s_%s_test.npy" % (datadir, dataset_name, sigma_dict[dataset_name])
            if os.path.exists(fname_rbf):
                k_test_X = np.load(fname_rbf)
            else:
                k_test_X = rbf_kernel(X_test, X_train, sigma_dict[dataset_name])
                # hu.save_pkl(fname_rbf, k_test_X)
                np.save(fname_rbf, k_test_X)
                print('%s saved' % fname_rbf)

            X_test = k_test_X
            X_test = torch.FloatTensor(X_test)
            Y_test = torch.FloatTensor(Y_test)

            dataset = torch.utils.data.TensorDataset(X_test, Y_test)

    if dataset_name == "matrix_fac":
        fname = datadir + 'matrix_fac.pkl'
        if not os.path.exists(fname):
            data = generate_synthetic_matrix_factorization_data()
            hu.save_pkl(fname, data)

        A, y = hu.load_pkl(fname)

        X_train, X_test, y_train, y_test = train_test_split(A, y, test_size=0.2, random_state=9513451)

        training_set = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
        test_set = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))

        if train_flag:
            dataset = training_set
        else:
            dataset = test_set

    return DatasetWrapper(dataset)

class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.fstar_list = None
        
    def __getitem__(self, index):
        data, target = self.dataset[index]

        if self.fstar_list is not None:
            fstar_list = self.fstar_list[index]
        else:
            fstar_list = -1

        return {"images":data, 
                'labels':target, 
                'meta':{'indices':index, 'fstar':fstar_list}}

    def compute_fstar(self, model_func, loss_function, fname):
        if os.path.exists(fname):
            fstar_list = hu.load_pkl(fname)
        else:
            fstar_list = np.ones(len(self)) * -1

            for i in range(len(self)):
                batch = self[i]
                images, labels = batch['images'][None].cuda(), batch['labels'][None].cuda()

                model = model_func()
                opt = torch.optim.Adam(model.parameters())

                for j in range(10000):
                    opt.zero_grad()

                    closure = lambda : loss_function(model, images, labels, backwards=True)
                    loss = opt.step(closure).item()
                    
                    grad_current = sps.get_grad_list(model.parameters())
                    grad_norm = sps.compute_grad_norm(grad_current)

                    if np.isnan(loss):
                        print('nan')
                    # print(i, loss)
                    if grad_norm < 1e-6:
                        break
                    if j > 0 and abs(loss_old - loss) < 1e-6:
                        break
                    loss_old = loss
                print("%d/%d - converged:%d - %.6f"% (i, len(self), j, loss))
                fstar_list[i] = loss
            hu.save_pkl(fname, fstar_list)

        self.fstar_list = fstar_list
        
    def __len__(self):
        return len(self.dataset)


def generate_synthetic_matrix_factorization_data(xdim=6, ydim=10, nsamples=1000, A_condition_number=1e-10):
    """
    Generate a synthetic matrix factorization dataset as suggested by Ben Recht.
    See: https://github.com/benjamin-recht/shallow-linear-net/blob/master/TwoLayerLinearNets.ipynb.
    """
    Atrue = np.linspace(1, A_condition_number, ydim
       ).reshape(-1, 1) * np.random.rand(ydim, xdim)
    # the inputs
    X = np.random.randn(xdim, nsamples)
    # the y's to fit
    Ytrue = Atrue.dot(X)
    data = (X.T, Ytrue.T)

    return data


def load_libsvm(name, data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    fn = LIBSVM_DOWNLOAD_FN[name]
    data_path = os.path.join(data_dir, fn)

    if not os.path.exists(data_path):
        url = urllib.parse.urljoin(LIBSVM_URL, fn)
        print("Downloading from %s" % url)
        urllib.request.urlretrieve(url, data_path)
        print("Download complete.")

    X, y = load_svmlight_file(data_path)
    return X, y

def rbf_kernel(A, B, sigma):
    # numpy version
    distsq = np.square(metrics.pairwise.pairwise_distances(A, B, metric="euclidean"))
    K = np.exp(-1 * distsq/(2*sigma**2))
    return K
