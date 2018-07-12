

import numpy as np
import os, sys, math
import datetime, pathlib

import torch
from torch.autograd import Variable
from torch import nn

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.stats import norm


def kl_loss(zmean, zlsig):
    b, l = zmean.size()

    kl = 0.5 * torch.sum(zlsig.exp() - zlsig + zmean.pow(2) - 1, dim=1)

    assert kl.size() == (b,)

    return kl

def sample(zmean, zlsig, eps=None):
    b, l = zmean.size()

    if eps is None:
        eps = torch.randn(b, l)
        if torch.cuda.is_available():
            eps = eps.cuda()
        eps = Variable(eps)

    return zmean + eps * (zlsig * 0.5).exp()


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def ensure(dir):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

def plot(latents, images, size=0.00001, filename='latent_space.pdf', invert=False):

    assert(latents.shape[0] == images.shape[0])

    mn, mx = np.min(latents), np.max(latents)

    n, h, w, c = images.shape

    aspect = h/w

    fig = plt.figure(figsize=(64,64))
    ax = fig.add_subplot(111)

    for i in range(n):
        x, y = latents[i, 0:2]

        im = images[i, :]

        ax.imshow(im if c > 1 else im.squeeze(0), extent=(x, x + size, y, y + size*aspect), cmap='gray_r' if invert else 'gray')

    # ax.scatter(latents[:, 0], latents[:, 1], alpha=0.5, linewidth=0)

    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)

    plt.savefig(filename)
    plt.close(fig)



class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class Debug(nn.Module):
    """
    Executes a lambda function and then returns the input. Useful for debugging.
    """
    def __init__(self, lambd):
        super(Debug, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        self.lambd(x)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view( (input.size(0),) + self.shape)


class Block(nn.Module):

    def __init__(self, in_channels, channels, num_convs = 3, kernel_size = 3, batch_norm=False, use_weight=True, use_res=True, deconv=False):
        super().__init__()

        self.layers = nn.Sequential()
        self.use_weight = use_weight
        self.use_res = use_res

        padding = int(math.floor(kernel_size / 2))

        self.upchannels = module=nn.Conv2d(in_channels, channels, kernel_size=1)

        for i in range(num_convs):
            if deconv:
                self.layers.add_module(name='deconv{:01}'.format(i), module=nn.ConvTranspose2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=not batch_norm))
            else:
                self.layers.add_module(name='conv{:01}'.format(i), module=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=not batch_norm))

            if batch_norm:
                self.layers.add_module(name='bn', module=nn.BatchNorm2d(channels))

            self.layers.add_module(name='act', module=nn.ReLU())

        if use_weight:
            self.weight = nn.Parameter(torch.randn(1))

    def forward(self, x):

        x = self.upchannels(x)

        out = self.layers(x)

        if not self.use_res:
            return out

        if not self.use_weight:
            return out + x

        return out + self.weight * x

