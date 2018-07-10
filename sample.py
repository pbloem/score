
from tqdm import tqdm
import os, random

import matplotlib
matplotlib.use('Agg')

from argparse import ArgumentParser

import pandas as pd
import wget
import numpy as np
import tqdm
import skvideo.io

from scipy.misc import imresize, imsave

def go(options):

    """Samples a small number of random frames from a large number of random videos"""

    # Set random or det. seed
    if options.seed < 0:
        seed = random.randint(0, 1000000)
    else:
        seed = options.seed

    np.random.seed(seed)
    print('random seed: ', seed)

    #- data urls
    df = pd.read_csv(options.video_urls, header=None)
    l = len(df)

    if options.num_videos is not None:

        rand_indices = random.sample(range(l), options.num_videos)
        urls = df.iloc[rand_indices, 2]
        num_videos = options.num_videos

    else:

        urls = df.iloc[:, 2]
        num_videos = len(df)

    ttl = num_videos * options.num_frames
    result = np.zeros(shape=(ttl, options.height, options.width, 3))

    failed_downloads = 0
    failed_reads = 0
    i = 0
    for url in tqdm.tqdm(urls):
        #- download videos. One for each instance in the batch.

        print('Downloading video', url)
        try:
            file = wget.download(url, out=options.data_dir)
        except Exception as e:
            print('*** Could not download', url, e)
            failed_downloads += 1
            continue

        try:
            gen = skvideo.io.vreader(file)

            length = 0
            for _ in gen:
                length += 1

        except Exception as e:
            print('*** Could not read video file ', url, e)
            failed_reads += 1
            continue

        print('\nlength', length)

        gen = skvideo.io.vreader(file, num_frames=length)

        frames = random.sample(range(length), options.num_frames)

        for f, frame in enumerate(gen):
            if f in frames:

                newsize = (options.height, options.width)
                frame = imresize(frame, newsize)/255

                result[i, ...] = frame
                i += 1

        os.remove(file)

    result = result[:i+1, ...]

    print('Sampling finished. Shape of final dataset:', result.shape)
    print('Number of download failures', failed_downloads)
    print('Number of file read failures', failed_reads)

    np.savez_compressed(options.result_dir + os.sep + options.result_name, images=result)

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-v", "--num-vids",
                        dest="num_videos",
                        help="Number of of videos to download.",
                        default=None, type=int)

    parser.add_argument("-f", "--frames",
                        dest="num_frames",
                        help="Number of frames to extract per video",
                        default=10, type=int)

    parser.add_argument("-W", "--width",
                        dest="width",
                        help="Width.",
                        default=320, type=int)

    parser.add_argument("-H", "--height",
                        dest="height",
                        help="Height.",
                        default=256, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random. Chosen seed will be printed to sysout",
                        default=1, type=int)

    parser.add_argument("-V", "--video-urls",
                        dest="video_urls",
                        help="CSV file with the video metadata",
                        default='./openbeelden.csv', type=str)

    parser.add_argument("-D", "--data-directory",
                        dest="data_dir",
                        help="Data directory",
                        default='./data', type=str)

    parser.add_argument("-R", "--result-directory",
                        dest="result_dir",
                        help="Result directory",
                        default='.', type=str)

    parser.add_argument("-N", "--result-filename",
                        dest="result_name",
                        help="Result filename",
                        default='sample.npz', type=str)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)