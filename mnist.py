"""
Simple MNIST sanity check for the VAE

"""


import torch

from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d, Linear, Sequential, ReLU, Sigmoid, Upsample
from torch.autograd import Variable
import torch.distributions as dist

import numpy as np
import ptutil, random, tqdm, math

from argparse import ArgumentParser

from tensorboardX import SummaryWriter

import torchvision
from torchvision import transforms

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def go(options):


    tbw = SummaryWriter(log_dir=options.tb_dir)

    # Set random or det. seed
    if options.seed < 0:
        seed = random.randint(0, 1000000)
    else:
        seed = options.seed

    np.random.seed(seed)
    print('random seed: ', seed)

    # load data

    normalize = transforms.Compose([transforms.ToTensor()])
    train = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=normalize)
    trainloader = torch.utils.data.DataLoader(train, batch_size=options.batch_size, shuffle=True, num_workers=2)

    test = torchvision.datasets.MNIST(root='./mnist', train=False, download=True, transform=normalize)
    testloader = torch.utils.data.DataLoader(test, batch_size=options.batch_size, shuffle=False, num_workers=2)

    ## Load the complete test set into a listr of batches
    test_batches = [inputs for inputs, _ in testloader]
    test_images = torch.cat(test_batches, dim=0).numpy()

    test_batch = test_batches[0]
    assert test_batch.size(0) > 10
    if torch.cuda.is_available():
        test_batch = test_batch.cuda()

    ## Build model

    # - channel sizes
    a, b, c = 8, 32, 128

    encoder = Sequential(
        Conv2d(1, a, (3, 3), padding=1), ReLU(),
        MaxPool2d((2, 2)),
        Conv2d(a, b, (3, 3), padding=1), ReLU(),
        MaxPool2d((2, 2)),
        Conv2d(b, c, (3, 3), padding=1), ReLU(),
        MaxPool2d((2, 2)),
        ptutil.Flatten(),
        Linear(3 * 3 * c, 2 * options.latent_size)
    )

    upmode = 'bilinear'
    decoder = Sequential(
        Linear(options.latent_size, c * 3 * 3), ReLU(),
        ptutil.Reshape((c, 3, 3)),
        Upsample(scale_factor=2, mode=upmode),
        ConvTranspose2d(c, b, (3, 3), padding=1), ReLU(),
        Upsample(scale_factor=2, mode=upmode),
        ConvTranspose2d(b, a, (3, 3), padding=0), ReLU(),
        Upsample(scale_factor=2, mode=upmode),
        ConvTranspose2d(a, 2, (3, 3), padding=1), Sigmoid()
    )

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    params = list(encoder.parameters()) + list(decoder.parameters())

    ### Fit model
    instances_seen = 0

    optimizer = Adam(params, lr=options.lr)

    for epoch in range(options.epochs):
        for i, data in tqdm.tqdm(enumerate(trainloader, 0)):
            # if i > 5:
            #     break

            # get the inputs
            inputs, labels = data

            b, c, w, h = inputs.size()

            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # wrap them in Variables
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            # Forward pass

            zcomb = encoder(inputs)
            zmean, zlsig = zcomb[:, :options.latent_size], zcomb[:, options.latent_size:]

            kl_loss = ptutil.kl_loss(zmean, zlsig)
            zsample = ptutil.sample(zmean, zlsig)

            out = decoder(zsample)

            m = dist.Normal(out[:, :1, :, :], out[:, 1:, :, :])
            rec_loss = - m.log_prob(inputs).view(b, -1).sum(dim=1)

            # rec_loss = binary_cross_entropy(out, inputs, reduce=False).view(b, -1).sum(dim=1)

            # Backward pass

            loss = (rec_loss + kl_loss).mean()
            loss.backward()

            optimizer.step()

            instances_seen += inputs.size(0)

            tbw.add_scalar('score/kl', float(kl_loss.mean()), instances_seen)
            tbw.add_scalar('score/rec', float(rec_loss.mean()), instances_seen)
            tbw.add_scalar('score/loss', float(loss), instances_seen)

        ## Plot some reconstructions
        if epoch % options.out_every == 0:
            print('({}) Plotting reconstructions.'.format(epoch))

            plt.figure(figsize=(10, 4))

            zc = encoder(test_batch)
            zmean, zlsig = zc[:, :options.latent_size], zc[:, options.latent_size:]
            zsample = ptutil.sample(zmean, zlsig)

            out = decoder(zsample)

            m = dist.Normal(out[:, :1, :, :], out[:, 1:, :, :])
            res = m.sample()
            res = res.clamp(0, 1)

            for i in range(10):
                ax = plt.subplot(4, 10, i + 1)
                ax.imshow(test_batch[i, :, :, :].cpu().squeeze(), cmap='gray')
                ptutil.clean(ax)

                ax = plt.subplot(4, 10, i + 11)
                ax.imshow(res[i, :, :, :].cpu().squeeze(), cmap='gray')
                ptutil.clean(ax)

                ax = plt.subplot(4, 10, i + 21)
                ax.imshow(out[i, :1, :, :].data.cpu().squeeze(), cmap='gray')
                ptutil.clean(ax)

                ax = plt.subplot(4, 10, i + 31)
                ax.imshow(out[i, 1:, :, :].data.cpu().squeeze(), cmap='gray')
                ptutil.clean(ax)


            # plt.tight_layout()
            plt.savefig('rec.{:03}.pdf'.format(epoch))

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=20, type=int)

    parser.add_argument("-o", "--out-every",
                        dest="out_every",
                        help="Output every x epochs.",
                        default=1, type=int)

    parser.add_argument("-L", "--latent-size",
                        dest="latent_size",
                        help="Size of the latent representation",
                        default=2, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="Batch size",
                        default=32, type=int)

    parser.add_argument("-T", "--tb-directory",
                        dest="tb_dir",
                        help="Tensorboard directory",
                        default='./runs/score/mnist', type=str)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random. Chosen seed will be printed to sysout",
                        default=-1, type=int)


    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)