from tqdm import tqdm
import math, sys, os, random
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from argparse import ArgumentParser

import torch
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy, relu
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
        return min(1, step / (total*1.5))

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
    p, q, r = 128, 64, 32


    encoder = Sequential(
        util.Block(3, a, use_res=options.use_res, batch_norm=options.use_bn),
        MaxPool2d((4,4)),
        util.Block(a, b, use_res=options.use_res, batch_norm=options.use_bn),
        MaxPool2d((4, 4)),
        util.Block(b, c, use_res=options.use_res, batch_norm=options.use_bn),
        MaxPool2d((4, 4)),
        util.Flatten(),
        Linear((WIDTH/64) * (HEIGHT/64) * c, p)
    )

    enc_dense1 = Linear(p, q)
    enc_dense2 = Linear(q, r)
    enc_dense3 = Linear(r, options.latent_size * 2)

    dec_dense1 = Linear(options.latent_size, r)
    dec_dense2 = Linear(r, q)
    dec_dense3 = Linear(q, p)

    upmode = 'bilinear'
    decoder = Sequential(
        Linear(p, 5 * 4 * c), ReLU(),
        # Linear(r, q), ReLU(),
        # Linear(q, p), ReLU(),
        # Linear(p,  5 * 4 * c), ReLU(),
        util.Reshape((c, 4, 5)),
        Upsample(scale_factor=4, mode=upmode),
        # util.Debug(lambda x: print('1', x.shape)),
        util.Block(c, c, deconv=True, use_res=options.use_res, batch_norm=options.use_bn),
        Upsample(scale_factor=4, mode=upmode),
        util.Block(c, b, deconv=True, use_res=options.use_res, batch_norm=options.use_bn),
        Upsample(scale_factor=4, mode=upmode),
        util.Block(b, a, deconv=True, use_res=options.use_res, batch_norm=options.use_bn),
        ConvTranspose2d(a, 3, kernel_size=1, padding=0),
        Sigmoid()
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

    for e in range(options.epochs):

        weight = 1 - anneal(e, options.epochs)
        print('epoch {}, weight {:.4}'.format(e, weight))

        for batch, _ in tqdm.tqdm(dataloader):

            if torch.cuda.is_available():
                batch = batch.cuda()
            batch = Variable(batch)

            optimizer.zero_grad()

            #- forward pass

            b, c, h, w = batch.size()

            xp = encoder(batch)

            xq    = relu(enc_dense1(xp))
            xr    = relu(enc_dense2(xq))
            zcomb = relu(enc_dense3(xr))

            zmean, zlsig = zcomb[:, :options.latent_size], zcomb[:, options.latent_size:]

            kl_loss = util.kl_loss(zmean, zlsig)

            zsample = util.sample(zmean, zlsig)

            xr_dec = relu(dec_dense1(zsample)) + weight * xr
            xq_dec = relu(dec_dense2(xr_dec))  + weight * xq
            xp_dec = relu(dec_dense3(xq_dec))  + weight * xp

            out = decoder(xp_dec)

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


    parser.add_argument("--res",
                        dest="use_res",
                        help="Whether to use residual connections.",
                        action="store_true")

    parser.add_argument("--bn",
                        dest="use_bn",
                        help="Whether to us batch normalization.",
                        action="store_true")

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)