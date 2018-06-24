from keras.utils import multi_gpu_model

import tensorflow as tf

from sklearn import datasets

from tqdm import tqdm
import math, sys, os, random
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from argparse import ArgumentParser

from keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Reshape, MaxPooling2D, UpSampling2D, Flatten, Cropping2D
from keras.models import Model, Sequential
from keras.engine.topology import Layer
from keras.utils import to_categorical
from keras import metrics
import keras.optimizers
from tensorflow.python.client import device_lib
import keras.backend as K

import numpy as np

from tensorboardX import SummaryWriter

import pandas as pd
import wget
import numpy as np
import os
import matplotlib.pyplot as plt
import tqdm, cv2

import util

WIDTH, HEIGHT = 320, 256

def anneal(step, total, k=1.0, anneal_function='logistic'):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - total / 2))))
    elif anneal_function == 'linear':
        return min(1, step / total)

def rec_loss(y_true, y_pred):
    reshape = Reshape((-1, WIDTH * HEIGHT * 3))
    y_true, y_pred = reshape(y_true), reshape(y_pred)

    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

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

    ## Build the model

    input = Input(shape=(HEIGHT, WIDTH, 3))

    a, b, c = 8, 32, 128

    h = Conv2D(a, (5, 5), activation='relu', padding='same')(input)
    h = Conv2D(a, (5, 5), activation='relu', padding='same')(h)
    h = Conv2D(a, (5, 5), activation='relu', padding='same')(h)
    h = MaxPooling2D((4, 4), padding='same')(h)

    h = Conv2D(b, (5, 5), activation='relu', padding='same')(h)
    h = Conv2D(b, (5, 5), activation='relu', padding='same')(h)
    h = Conv2D(b, (5, 5), activation='relu', padding='same')(h)
    h = MaxPooling2D((4, 4), padding='same')(h)

    h = Conv2D(c, (5, 5), activation='relu', padding='same')(h)
    h = Conv2D(c, (5, 5), activation='relu', padding='same')(h)
    h = Conv2D(c, (5, 5), activation='relu', padding='same')(h)
    h = MaxPooling2D((4, 4), padding='same')(h)

    h = Flatten()(h)
    zmean = Dense(options.latent_size)(h)
    zlsig = Dense(options.latent_size)(h)

    kl = util.KLLayer()
    [zmean, zlsig] = kl([zmean, zlsig])
    zsample = util.Sample()([zmean, zlsig])

    h = Dense(5 * 4 * 128, activation='relu')(zsample)
    #  h = Dense(HEIGHT//(4*4*4) * WIDTH//(4*4*4) * 128, activation='relu')(zsample)

    h = Reshape((4, 5, 128))(h)
    h = UpSampling2D((4, 4))(h)

    h = Conv2DTranspose(c, (5, 5), activation='relu', padding='same')(h)
    h = Conv2DTranspose(c, (5, 5), activation='relu', padding='same')(h)
    h = Conv2DTranspose(c, (5, 5), activation='relu', padding='same')(h)
    h = UpSampling2D((4, 4))(h)

    h = Conv2DTranspose(b, (5, 5), activation='relu', padding='same')(h)
    h = Conv2DTranspose(b, (5, 5), activation='relu', padding='same')(h)
    h = Conv2DTranspose(b, (5, 5), activation='relu', padding='same')(h)
    h = UpSampling2D((4, 4))(h)

    h = Conv2DTranspose(a, (5, 5), activation='relu', padding='same')(h)
    h = Conv2DTranspose(a, (5, 5), activation='relu', padding='same')(h)
    output = Conv2DTranspose(3, (5, 5), activation='sigmoid', padding='same')(h)

    encoder = Model(input, [zmean, zlsig])
    # decoder = Model(zsample, output)
    auto = Model(input, output)

    if options.num_gpu is not None:
        auto = multi_gpu_model(auto, gpus=options.num_gpu)

    opt = keras.optimizers.Adam(lr=options.lr)
    auto.compile(optimizer=opt,
                 loss=rec_loss)

    ## Training loop

    #- data urls
    df = pd.read_csv(options.video_urls, header=None)

    # Test images to plot
    images = np.load(options.sample_file)['images']

    for ep in tqdm.trange(options.num_videos):

        print('Set KL weight to ', anneal(ep, options.num_videos))
        K.set_value(kl.weight, anneal(ep, options.num_videos))

        #- download videos. One for each instance in the batch.
        l = len(df)
        rand_indices = random.sample(range(l), options.batch_size)

        caps = [] # video capture objects
        files = []

        if options.model_file is not None:
            auto.save(options.model_file)

        print('downloading video batch.')
        for url in df.iloc[rand_indices, 2]:
            try:
                file = wget.download(url, out=options.data_dir)
            except:
                raise Exception('Download failed for file', url)
            caps.append(cv2.VideoCapture(file))
            files.append(file)
        print('done.')

        instances_seen = 0

        while True:
            finished = True

            batch = np.zeros(shape=(options.batch_size, HEIGHT, WIDTH, 3))


            for i, cap in enumerate(caps):
                if cap.isOpened():
                    ret, frame = cap.read()
                    if(ret):
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        newsize = (WIDTH, HEIGHT)
                        frame = cv2.resize(frame, newsize)
                        batch[i, ...] = frame
                        finished = False

            l = auto.train_on_batch(batch, batch)

            instances_seen += batch.shape[0]

            if l.squeeze().ndim == 0:
                tbw.add_scalar('score/sum', float(l), instances_seen)
            else:
                tbw.add_scalar('score/sum', float(np.sum(l) / len(l)), instances_seen)

            if finished:
                break

        for cap in caps:
            cap.release()

        for file in files:
            os.remove(file)

        ## Plot the latent space
        if options.sample_file is not None:
            print('Plotting latent space.')

            latents = encoder.predict(images)[0]
            print('-- Computed latent vectors.')

            rng = np.max(latents[:, 0]) - np.min(latents[:, 0])

            print('-- L', latents[:10,:])
            print('-- range', rng)

            n = latents.shape[0]
            util.plot(latents, images, size=rng/math.sqrt(n), filename='score.{:04}.pdf'.format(e))




if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-v", "--num-vid-batches",
                        dest="num_videos",
                        help="Number of of video batches.",
                        default=150, type=int)

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
                        default=8, type=int)

    parser.add_argument("-D", "--data-directory",
                        dest="data_dir",
                        help="Data directory",
                        default='./data', type=str)

    parser.add_argument("-T", "--tb-directory",
                        dest="tb_dir",
                        help="Tensorboard directory",
                        default='./runs/lm', type=str)

    parser.add_argument("-M", "--model-file",
                        dest="model_file",
                        help="Where to save the model (if None, the model will not be saved). The model will be overwritten every video batch.",
                        default=None, type=str)

    parser.add_argument("-V", "--video-urls",
                        dest="video_urls",
                        help="CSV file with the video metadata",
                        default='./openbeelden.csv', type=str)

    parser.add_argument("-S", "--sample-file",
                        dest="sample_file",
                        help="Saved numpy array with random frames",
                        default=None, type=str)


    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random. Chosen seed will be printed to sysout",
                        default=1, type=int)

    parser.add_argument("-g", "--num-gpu",
                        dest="num_gpu",
                        help="How many GPUs to use",
                        default=None, type=int)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)