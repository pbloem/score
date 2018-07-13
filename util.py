

import numpy as np
import os, sys, math
import datetime, pathlib
import random

import torch
from torch.autograd import Variable
from torch import nn

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec

from scipy.stats import norm

def clean(axes=None):

    if axes is None:
        axes = plt.gca()

    axes.spines["right"].set_visible(False)
    axes.spines["top"].set_visible(False)
    axes.spines["bottom"].set_visible(False)
    axes.spines["left"].set_visible(False)

    axes.get_xaxis().set_tick_params(which='both', top='off', bottom='off', labeltop='off', labelbottom='off')
    axes.get_yaxis().set_tick_params(which='both', left='off', right='off', labelleft='off', labelright='off')

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

def interpolate(images, encoder, decoder, steps=7, name='interpolate', mode='spherical', reps=5):
    """
    Plots a grid of values interpolating (linearly)  between four given items.

    :param name:
    :return:
    """

    plt.figure(figsize=(steps + 2, reps))

    f, aa = plt.subplots(reps, steps + 2, gridspec_kw = {'wspace':0, 'hspace':0.01})

    for rep in range(reps):

        x1 = images[random.randint(0, images.size(0))]
        x2 = images[random.randint(0, images.size(0))]

        x1, x2 = x1.unsqueeze(0).float(), x2.unsqueeze(0).float()

        if torch.cuda.is_available():
            x1, x2 = x1.cuda(), x2.cuda()
        x1, x2 = Variable(x1), Variable(x2)

        z1, z2 = encoder(x1), encoder(x2)

        if mode == 'spherical':
            zs = slerp(z1, z2, steps)
        elif mode == 'linear':
            zs = linp(z1, z2, steps)
        else:
            raise Exception('Mode {} not recognized'.format(mode))

        out = decoder(zs).data
        out = np.transpose(out.cpu().numpy(), (0, 2, 3, 1))

        for i in range(steps):
            aa[rep, i+1].imshow(out[i])

        aa[rep,  0].imshow(np.transpose(x1[0].cpu().numpy(), (1, 2, 0)))
        aa[rep, -1].imshow(np.transpose(x2[0].cpu().numpy(), (1, 2, 0)))

    for i in range(aa.shape[0]):
        for j in range(aa.shape[1]):
            clean(aa[i,j])

    plt.savefig(name + '.pdf')

def linp(x, y, steps=5):
    """
    Produces a spherical linear interpolation between two points

    :param x:
    :param y:
    :param steps:
    :return:
    """
    assert x.size(0) == y.size(0)
    n = x.size(0)

    d = torch.linspace(0, 1, steps)

    return x.unsqueeze(0).expand(steps, n) * d.unsqueeze(1) \
           + y.unsqueeze(0).expand(steps, n) * (1-d).unsqueeze(1)

def slerp(x, y, steps=5):
    """
    Produces a spherical linear interpolation between two points

    :param x:
    :param y:
    :param steps:
    :return:
    """
    assert x.size(0) == y.size(0)

    x, y = x[0], y[0]

    n = x.size(0)

    angle = torch.acos(torch.dot(x, y)/(x.norm() * y.norm()))

    d = torch.linspace(0, 1, steps).unsqueeze(1)

    if torch.cuda.is_available():
        d     = d.cuda()
        angle = angle.cuda()

    d1 = torch.sin(d     * angle) / torch.sin(angle)
    d2 = torch.sin((1-d) * angle) / torch.sin(angle)

    return   x.unsqueeze(0).expand(steps, n) * d1 \
           + y.unsqueeze(0).expand(steps, n) * d2




