from tqdm import tqdm
import math, sys, os, random
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from argparse import ArgumentParser

import torch
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d, Linear, Sequential, ReLU, Sigmoid, Upsample
from torch.autograd import Variable

from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision import transforms

import numpy as np

from tensorboardX import SummaryWriter

from scipy.misc import imresize

import pandas as pd
import wget
import numpy as np
import os, time
import matplotlib.pyplot as plt
import tqdm

import skvideo.io

import util

WIDTH, HEIGHT = 320, 256

def anneal(step, total, k=1.0, anneal_function='logistic'):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - total / 2))))
    elif anneal_function == 'linear':
        return min(1, step / total)

def go(options):

    ## Admin

    # Tensorboard output
    tbw = SummaryWriter(log_dir=options.tb_dir)

    # Set random or det. seed
    if options.seed < 0:
        seed = random.randint(0, 1000000)
    else:
        seed = options.seed

    np.random.seed(seed)
    print('random seed: ', seed)

    ## Load the data
    transform = transforms.Compose([transforms.ToTensor()])


    data = torchvision.datasets.ImageFolder(options.data_dir, transform=transform)


    dataloader = torch.utils.data.DataLoader(data,
                                              batch_size=options.batch_size,
                                              shuffle=True,
                                              num_workers=2)

    ## Build the model

    #- channel sizes
    a, b, c = 8, 32, 128

    encoder = Sequential(
        Conv2d(3, a, (5, 5), padding=2), ReLU(),
        Conv2d(a, a, (5, 5), padding=2), ReLU(),
        Conv2d(a, a, (5, 5), padding=2), ReLU(),
        MaxPool2d((4,4)),
        Conv2d(a, b, (5, 5), padding=2), ReLU(),
        Conv2d(b, b, (5, 5), padding=2), ReLU(),
        Conv2d(b, b, (5, 5), padding=2), ReLU(),
        MaxPool2d((4, 4)),
        Conv2d(b, c, (5, 5), padding=2), ReLU(),
        Conv2d(c, c, (5, 5), padding=2), ReLU(),
        Conv2d(c, c, (5, 5), padding=2), ReLU(),
        MaxPool2d((4, 4)),
        util.Flatten(),
        Linear((WIDTH/64) * (HEIGHT/64) * c, 2 * options.latent_size)
    )

    upmode = 'bilinear'
    decoder = Sequential(
        Linear(options.latent_size, 5 * 4 * c), ReLU(),
        util.Reshape((c, 4, 5)),
        Upsample(scale_factor=4, mode=upmode),
        # util.Debug(lambda x: print('1', x.shape)),
        ConvTranspose2d(c, c, (5, 5), padding=2), ReLU(),
        ConvTranspose2d(c, c, (5, 5), padding=2), ReLU(),
        ConvTranspose2d(c, b, (5, 5), padding=2), ReLU(),
        Upsample(scale_factor=4, mode=upmode),
        ConvTranspose2d(b, b, (5, 5), padding=2), ReLU(),
        ConvTranspose2d(b, b, (5, 5), padding=2), ReLU(),
        ConvTranspose2d(b, a, (5, 5), padding=2), ReLU(),
        Upsample(scale_factor=4, mode=upmode),
        ConvTranspose2d(a, a, (5, 5), padding=2), ReLU(),
        ConvTranspose2d(a, a, (5, 5), padding=2), ReLU(),
        ConvTranspose2d(a, 3, (5, 5), padding=2), Sigmoid()
    )

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    ## Training loop

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = Adam(params, lr=options.lr)

    ### Fit model
    instances_seen = 0

    # Test images to plot
    images = torch.from_numpy(np.load(options.sample_file)['images']).permute(0, 3, 1, 2)

    for e in tqdm.trange(options.epochs):
        for batch, _ in tqdm.tqdm(dataloader):

            if torch.cuda.is_available():
                batch = batch.cuda()
            batch = Variable(batch)

            optimizer.zero_grad()

            #- forward pass

            b, c, h, w = batch.size()

            zcomb = encoder(batch)
            zmean, zlsig = zcomb[:, :options.latent_size], zcomb[:, options.latent_size:]

            kl_loss = util.kl_loss(zmean, zlsig)

            zsample = util.sample(zmean, zlsig)

            out = decoder(zsample)

            rec_loss = binary_cross_entropy(out, batch, reduce=False).view(b, -1).sum(dim=1)

            #- backward pass

            loss = (rec_loss + kl_loss).mean()
            loss.backward()

            optimizer.step()

            instances_seen += batch.size(0)

            tbw.add_scalar('score/kl', float(kl_loss.mean()), instances_seen)
            tbw.add_scalar('score/rec', float(rec_loss.mean()), instances_seen)
            tbw.add_scalar('score/loss', float(loss), instances_seen)

        ## Plot the latent space
        if options.sample_file is not None and e % options.out_every == 0:

            if options.model_dir is not None:
                torch.save(encoder.state_dict(),
                           options.model_dir + '/encoder.{}.{:.4}.model'.format(e, float(loss)))
                torch.save(decoder.state_dict(),
                           options.model_dir + '/decoder.{}.{:.4}.model'.format(e, float(loss)))

            print('Plotting latent space.')

            l = images.size(0)
            b = options.batch_size

            out_batches = []

            for fr in range(0, l, b):
                to = min(fr + b, l)

                batch = images[fr:to]

                if torch.cuda.is_available():
                    batch = batch.cuda()
                batch = Variable(batch)
                out = encoder(batch.float()).data[:, :options.latent_size]

                out_batches.append(out)

            latents = torch.cat(out_batches, dim=0)

            print('-- Computed latent vectors.')

            rng = float(torch.max(latents[:, 0]) - torch.min(latents[:, 0]))

            print('-- L', latents[:10, :])
            print('-- range', rng)

            n_test = latents.shape[0]
            util.plot(latents.cpu().numpy(), images.permute(0, 2, 3, 1).numpy(), size=rng / math.sqrt(n_test),
                      filename='score.{:04}.pdf'.format(e), invert=True)

            print('-- finished plot')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of 'epochs'.",
                        default=50, type=int)

    parser.add_argument("-o", "--out-every",
                        dest="out_every",
                        help="How many epochs to wait before producing output.",
                        default=1, type=int)

    parser.add_argument("-L", "--latent-size",
                        dest="latent_size",
                        help="Size of the latent representation",
                        default=256, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="Batch size",
                        default=32, type=int)

    parser.add_argument("-D", "--data-directory",
                        dest="data_dir",
                        help="Data directory",
                        default='./data', type=str)

    parser.add_argument("-T", "--tb-directory",
                        dest="tb_dir",
                        help="Tensorboard directory",
                        default='./runs/score', type=str)

    parser.add_argument("-M", "--model-dir",
                        dest="model_dir",
                        help="Where to save the model (if None, the model will not be saved). The model will be overwritten every video batch.",
                        default=None, type=str)

    parser.add_argument("-S", "--sample-file",
                        dest="sample_file",
                        help="Saved numpy array with random frames",
                        default='./sample.npz', type=str)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random. Chosen seed will be printed to sysout",
                        default=1, type=int)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)