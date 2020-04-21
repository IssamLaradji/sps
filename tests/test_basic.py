import sys, os
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

import unittest
import numpy as np 
import os
import torch
import shutil



from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
from haven import haven_jobs as hjb
import sps
import torch

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

class Test(unittest.TestCase):

    def test_update(self):
        torch.manual_seed(1)
        X = torch.randn(100,2)
        Y = torch.randn(100)

        model = linearRegression(2, 1)
        # opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        
        opt = sps.Sps(model.parameters())

        for epoch in range(10):
            opt.zero_grad()
            loss = torch.nn.MSELoss() (model(X), Y)
            loss.backward()

            opt.step(loss=loss)

            print('epoch {}, loss {}'.format(epoch, loss.item()))

        
if __name__ == '__main__':
    unittest.main()
