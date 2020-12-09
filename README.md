# onlinehd

**Authors**: Alejandro Hernández Cano, Mohsen Imani.

## Installation

In order to install the package, simply run the following:

```
pip install onlinehd
```

Visit the PyPI [project page](https://pypi.org/project/onlinehd/) for
more information about releases.

## Documentation

Read the [documentation](https://onlinehd.readthedocs.io/en/latest/)
of this project. 

## Quick start

The following code generates dummy data and trains a OnlnineHD classification
model with it.

```python
>>> import onlinehd
>>> dim = 10000
>>> n_samples = 1000
>>> features = 100
>>> classes = 5
>>> x = torch.randn(n_samples, features) # dummy data
>>> y = torch.randint(0, classes, [n_samples]) # dummy data
>>> model = onlinehd.OnlineHD(classes, features, dim=dim)
>>> if torch.cuda.is_available():
...     print('Training on GPU!')
...     model = model.to('cuda')
...     x = x.to('cuda')
...     y = y.to('cuda')
...
Training on GPU!
>>> model.fit(x, y, epochs=10)
>>> ypred = model(x)
>>> ypred.size()
torch.Size([1000])
```

For more examples, see the `example.py` script. Be aware that this script needs
`pytorch`, `sklearn` and `numpy` to run.

## Citation Request

If you use onlinehd code, please cite the following paper:

1. Alejandro Hernández-Cano, Namiko Matsumoto, Eric Ping, Mohsen Imani
   "OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using
   Hyperdimensional System", IEEE/ACM Design Automation and Test in Europe
   Conference (DATE), 2021.
