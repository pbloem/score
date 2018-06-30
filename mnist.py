"""
Simple MNIST sanity check for the VAE

"""


import torch

from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d, Linear, Sequential, ReLU, Sigmoid, Upsample
from torch.autograd import Variable

import numpy as np
import util, random, tqdm, math

from argparse import ArgumentParser

from tensorboardX import SummaryWriter

import torchvision
from torchvision import transforms

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

    ## Build model

    # - channel sizes
    a, b, c = 8, 32, 128

    encoder = Sequential(
        Conv2d(1, a, (5, 5), padding=2), ReLU(),
        Conv2d(a, a, (5, 5), padding=2), ReLU(),
        Conv2d(a, a, (5, 5), padding=2), ReLU(),
        MaxPool2d((2, 2)),
        Conv2d(a, b, (5, 5), padding=2), ReLU(),
        Conv2d(b, b, (5, 5), padding=2), ReLU(),
        Conv2d(b, b, (5, 5), padding=2), ReLU(),
        MaxPool2d((2, 2)),
        Conv2d(b, c, (5, 5), padding=2), ReLU(),
        Conv2d(c, c, (5, 5), padding=2), ReLU(),
        Conv2d(c, c, (5, 5), padding=2), ReLU(),
        MaxPool2d((2, 2)),
        util.Flatten(),
        Linear(1152, 2 * options.latent_size)
    )

    upmode = 'bilinear'
    decoder = Sequential(
        Linear(options.latent_size, 4 * 4 * c), ReLU(),
        util.Reshape((c, 4, 4)),
        ConvTranspose2d(c, c, (5, 5), padding=2), ReLU(),
        ConvTranspose2d(c, c, (5, 5), padding=2), ReLU(),
        ConvTranspose2d(c, b, (5, 5), padding=2), ReLU(),
        Upsample(scale_factor=3, mode=upmode),
        ConvTranspose2d(b, b, (5, 5), padding=2), ReLU(),
        ConvTranspose2d(b, b, (5, 5), padding=2), ReLU(),
        ConvTranspose2d(b, a, (5, 5), padding=2), ReLU(),
        Upsample(scale_factor=2, mode=upmode),
        ConvTranspose2d(a, a, (5, 5), padding=2), ReLU(),
        ConvTranspose2d(a, a, (5, 5), padding=2), ReLU(),
        ConvTranspose2d(a, 1, (5, 5), padding=0), Sigmoid()
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

            # get the inputs
            inputs, labels = data

            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # wrap them in Variables
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            # Forward pass
            eps = torch.randn(options.batch_size, options.latent_size)

            zcomb = encoder(inputs)
            zmean, zlsig = zcomb[:, :options.latent_size], zcomb[:, options.latent_size:]

            kl_loss = util.kl_loss(zmean, zlsig)

            zsample = util.sample(zmean, zlsig, eps)

            out = decoder(zsample)

            rec_loss = binary_cross_entropy(out, inputs)

            # Backward pass

            loss = (rec_loss + kl_loss).mean()
            loss.backward()

            optimizer.step()

            instances_seen += inputs.size(0)

            tbw.add_scalar('score/kl', float(kl_loss.mean()), instances_seen)
            tbw.add_scalar('score/rec', float(rec_loss.mean()), instances_seen)
            tbw.add_scalar('score/loss', float(loss), instances_seen)

        ## Plot the latent space
        if epoch % options.out_every == 0:
            print('Plotting latent space.')

            out_batches = [None] * test_batches
            for i, batch in enumerate(test_batches):
                if torch.cuda.is_available():
                    batch = batch.cuda()
                batch = Variable(batch)
                out_batches[i] = encoder(batch)[:options.latent_size].data

            latents = torch.cat(out_batches, dim=0)

            print('-- Computed latent vectors.')

            rng = np.max(latents[:, 0]) - np.min(latents[:, 0])

            print('-- L', latents[:10,:])
            print('-- range', rng)

            n_test = latents.shape[0]
            util.plot(latents.numpy(), test_images, size=rng/math.sqrt(n_test), filename='mnist.{:04}.pdf'.format(epoch), invert=True)
            print('-- finished plot')


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