## SPS - Stochastic Polyak Step-size [[paper]](https://arxiv.org/pdf/2002.10542.pdf)

Fast convergence with SPS optimizer. The first efficient stochastic variant of the classical Polyak step-size for SGD


### 1. Install requirements
Install the sps optimizer.

`pip install git+https://github.com/IssamLaradji/sps.git`


Install the [Haven library](https://github.com/ElementAI/haven) for managing the experiments.

`pip install -r requirements.txt`

### 2. Usage
Use `Sps` in your code by adding the following script.

```python
import sps
opt = sps.Sps(model.parameters())

for epoch in range(100):
    opt.zero_grad()
    loss = torch.nn.MSELoss() (model(X), Y)
    loss.backward()

    opt.step(loss=loss)
```

### 3. Mnist experiment

`python trainval.py -e mnist -sb ../results -r 1`

where `-e` is the experiment group, `-sb` is the result directory.

### 4. kernel experiment

`python trainval.py -e kernel -sb ../results -r 1`


### 5. Visualize

Create the jupyter by running `create_jupyter.py`, and run the first cell to get plots, 

![alt text](results/results.png)


#### Citation

```
@article{loizou2020stochastic,
  title={Stochastic polyak step-size for SGD: An adaptive learning rate for fast convergence},
  author={Loizou, Nicolas and Vaswani, Sharan and Laradji, Issam and Lacoste-Julien, Simon},
  journal={arXiv preprint arXiv:2002.10542},
  year={2020}
}
```

It is a collaborative work between labs at MILA, Element AI, and UBC.
