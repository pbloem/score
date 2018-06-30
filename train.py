from tqdm import tqdm
import math, sys, os, random
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from argparse import ArgumentParser

import torch
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d, Linear, Sequential, ReLU, Upsample

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

    util.ensure(options.data_dir)

    ## Download videos
    df = pd.read_csv(options.video_urls, header=None)
    if options.max_videos is not None:
        df = df[:options.max_videos]

    urls = df.iloc[:, 2]

    files = []
    lengths = []

    t0 = time.time()

    for url in tqdm.tqdm(urls):
        # - download videos. One for each instance in the batch.

        print('Downloading video', url)
        file = wget.download(url, out=options.data_dir)

        gen = skvideo.io.vreader(file)

        length = 0
        for _ in gen:
            length += 1

        if length > 100:
            files.append(file)
            lengths.append(length)

    print('All {} videos downloaded ({} s)'.format(len(files), time.time()-t0))
    print('Total number of frames in data:', sum(lengths))

    ## Build the model

    #- channel sizes
    a, b, c = 8, 32, 128

    encoder = Sequential([
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
        Linear(-1, 2 * options.latent_size)
    ])


    upmode = 'bilinear'
    decoder = Sequential([
        Linear(options.latent_size, 5 * 4 * c), ReLU(),
        util.Reshape((4, 5, c)),
        Upsample(scale_factor=4, mode=upmode),
        ConvTranspose2d(c, c, (5, 5), padding=2), ReLU(),
        ConvTranspose2d(c, c, (5, 5), padding=2), ReLU(),
        ConvTranspose2d(c, c, (5, 5), padding=2), ReLU(),
        Upsample(scale_factor=4, mode=upmode),
        ConvTranspose2d(b, b, (5, 5), padding=2), ReLU(),
        ConvTranspose2d(b, b, (5, 5), padding=2), ReLU(),
        ConvTranspose2d(b, b, (5, 5), padding=2), ReLU(),
        Upsample(scale_factor=4, mode=upmode),
        ConvTranspose2d(a, a, (5, 5), padding=2), ReLU(),
        ConvTranspose2d(a, a, (5, 5), padding=2), ReLU(),
        ConvTranspose2d(a, 3, (5, 5), padding=2), ReLU(),
    ])

    ## Training loop

    # Test images to plot
    images = np.load(options.sample_file)['images']

    instances_seen = 0
    per_video = math.ceil(options.epoch_size / len(files))
    total = len(files) * per_video

    for ep in tqdm.trange(options.epochs):
        print('Sampling batch'); t0 = time.time()

        batch = np.zeros((total, HEIGHT, WIDTH, 3))
        i = 0

        for file, length in zip(files, lengths):

            gen = skvideo.io.vreader(file)
            frames = random.sample(range(length), per_video)

            for i, frame in enumerate(gen):
                try:
                    frame = next(gen)
                    if i in frames:
                        newsize = (HEIGHT, WIDTH)

                        frame = imresize(frame, newsize)/255

                        batch[i] = frame

                except Exception as e:
                    pass

        print('Batch sampled ({} s).'.format(time.time() - t0))
        print('Batch size:', batch.shape); t0 = time.time()

        eps = np.random.randn(batch.shape[0], options.latent_size)
        l = auto.fit([batch, eps], batch, epochs=1, validation_split=1/10, shuffle=True)
        print('Batch trained ({} s).'.format(time.time() - t0))

        instances_seen += batch.shape[0]

        # if l.squeeze().ndim == 0:
        #     l = float(l)
        # else:
        #     l = float(np.sum(l) / len(l))
        #
        # tbw.add_scalar('score/sum', l, instances_seen)

        if options.model_dir is not None:
            encoder.save(options.model_dir + '/encoder.{}.{:04}.keras_model'.format(ep, l))

        ## Plot the latent space
        if options.sample_file is not None:
            print('Plotting latent space.')

            latents = encoder.predict(images)[0]
            print('-- Computed latent vectors.')

            rng = np.max(latents[:, 0]) - np.min(latents[:, 0])

            n_test = latents.shape[0]

            print('-- L', latents[:10,:])
            print('-- range', rng)
            print('-- plot size', rng/math.sqrt(n_test))

            util.plot(latents, images, size=rng/math.sqrt(n_test), filename='score.{:04}.pdf'.format(ep))

    for file in files:
        os.remove(file)

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of 'epochs'.",
                        default=50, type=int)

    parser.add_argument("-E", "--epoch-size",
                        dest="epoch_size",
                        help="How many frames to sample for one 'epoch'.",
                        default=10000, type=int)

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

    parser.add_argument("-V", "--video-urls",
                        dest="video_urls",
                        help="CSV file with the video metadata",
                        default='./openbeelden.csv', type=str)

    parser.add_argument("-S", "--sample-file",
                        dest="sample_file",
                        help="Saved numpy array with random frames",
                        default='./sample.npz', type=str)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random. Chosen seed will be printed to sysout",
                        default=1, type=int)

    parser.add_argument("-m", "--max-videos",
                        dest="max_videos",
                        help="Limit the total number of videos analyzed. (If None all videos are downloaded).",
                        default=None, type=int)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)