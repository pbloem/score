import keras
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenetv2 import MobileNetV2

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image
import numpy as np

import magenta.music as mm
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel

import skvideo.io
import wget, tqdm, os, tfutil, sys

import pandas as pd

from scipy.io import wavfile

from argparse import ArgumentParser

from subprocess import call

# suppress warnings
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from sklearn.decomposition import PCA

from skimage.transform import resize

import random

"""
Generate a MIDI track from a movie file.

The pipeline used by this script works as follows.

1) A pretrained model (InceptionV3) is used to map the each frame in the video to a vector of _image features_.
2) A _mapper_ is used to convert each cluster of n frames to a latent vectors representing 16 bars of music. The mapper
ensures that the resulting latent variables fit the standard normal shape of the latent space. Two mappers are available:
 * PCA: This mapper only looks at the frames of the current video, and fits them to a standard normal distribution
using a Principal Component Analysis. This is the default mapper.
 * vae.???: This mapper is the encoder part of a VAE trained on image features extracted from the the whole openbeelden archive.
 Choose this mapper by adding the argument the argument "-m vae.??".
3) The resulting latent vectors are fed to the MusicVAE decoder to generate 16 bars of music.
4) The generated music is saved three ways: as a midi file, as a synthesized .wav file, as a .mkv file combining the
original video with the new music (only works if ffmpeg is installed). If the input video had audio, this is discarded.

## Notes

* In principle, any input video should work, but video loading can be tricky. For best results, use videos from the sound
 and vision "openbeelden" archive. The file openbeelden.clean.csv contains URLs for over 3000 videos that should work.
* The model is entirely unsupervised. The music will change in response to high level semantic features in the frames of the
video, but which frames correspond to which music features is entirely random.
* The PCA mapper will maximize the response to variance _within_ the same video. The pretrained mapper instead may
generate more homogeneous music for a single video, but is more likely to assign separate videos their own characteristic
music track.
** The model is trained on 2 bar chunks for the melody and drum models and 16 bar chunks for the poly model, so those
 so generating chunks of that length should provide the most natural results. However, shorter values provide a more
 direct response to what is happening in the video.
* (TODO) Any tensorflow model that maps a (b*???, ???)-tensor to a (b, 512) tensor can be used as a custom mapper. Just
save the model and load it with the "-m" switch. For good results, the model should map to points that are likely under
the multivariate standard normal distribution.
"""

FPS = 25             # We assume PAL for now
SECONDS_PER_BAR = 2  # Default MIDI timing (120BPM, 480 BPQ)
MVAE_URL_DRUMS = 'https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/cat-drums_2bar_small.lokl.tar'
MVAE_URL_MEL   = 'https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/cat-mel_2bar_big.tar'
MVAE_URL_POLY  = 'https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/hierdec-trio_16bar.tar'
SAMPLE_RATE = 44100
FRAMECHUNK = 100 # Set as big as memory allows

def go(arg):

    # Load pretrained models
    ## Load the Music VAE model
    if arg.decoder == 'melody':
        mfile = arg.model_dir + os.sep + 'musicmodel.melody.tar'
        if not os.path.isfile(mfile):
            print('Downloading MusicVAE (melody model).')
            wget.download(MVAE_URL_MEL, mfile)

        decoder_config = configs.CONFIG_MAP['cat-mel_2bar_big']
        decoder = TrainedModel(decoder_config, batch_size=4, checkpoint_dir_or_path=mfile)
        latent_size = 256

    elif arg.decoder == 'drums':
        mfile = arg.model_dir + os.sep + 'musicmodel.drums.tar'
        if not os.path.isfile(mfile):
            print('Downloading MusicVAE (drums model).')
            wget.download(MVAE_URL_DRUMS, mfile)

        decoder_config = configs.CONFIG_MAP['cat-drums_2bar_small']
        decoder = TrainedModel(decoder_config, batch_size=4, checkpoint_dir_or_path=mfile)
        latent_size = 128

    elif arg.decoder == 'poly':
        mfile = arg.model_dir + os.sep + 'musicmodel.poly.tar'
        if not os.path.isfile(mfile):
            print('Downloading MusicVAE (polyphonic model).')
            wget.download(MVAE_URL_POLY, mfile)

        decoder_config = configs.CONFIG_MAP['hierdec-trio_16bar']
        decoder = TrainedModel(decoder_config, batch_size=4, checkpoint_dir_or_path=mfile)
        latent_size = 256

    else:
        raise Exception('Decoder model {} not recognized. Use "poly", "melody" or "drums"'.format(arg.decoder))

    shape = None

    if arg.encoder == 'inceptionv3':
        encoder = InceptionV3(weights='imagenet', include_top=False)
        prep = keras.applications.inception_v3.preprocess_input
        flat = 6 * 8 * 2048
    elif arg.encoder == 'mobilenetv2':
        encoder = MobileNetV2(weights='imagenet', include_top=False)
        prep = keras.applications.mobilenetv2.preprocess_input
        flat = 7 * 7 * 1280
        shape = (224, 244)
    else:
        raise Exception('Encoder model {} not recognized'.format(arg.encoder))

    frames_per_chunk = arg.chunk_length * SECONDS_PER_BAR * FPS

    has_video = True

    if arg.input == 'none':
        ## Generate 6 bars of random music
        z = np.random.randn(6, latent_size)
        has_video = False

    elif arg.input == 'slerp':
        ## Generate 6 bars of random music
        z0 = np.random.randn(latent_size) * 2
        z1 = np.random.randn(latent_size) * 2

        z = tfutil.slerp(z0, z1, steps=10)

        has_video = False
    else:
        # Load a random video from the openbeelden data
        if arg.input == 'random':
            # - data urls
            df = pd.read_csv(tfutil.DIR + os.sep + 'openbeelden.clean.csv', header=None)
            l = len(df)

            index = random.randint(0, l)
            url = df.iloc[index, 2]

            print('Downloading video', url)
            try:
                dir = './downloaded/'
                tfutil.ensure(dir)

                arg.input = wget.download(url, out=dir)
            except Exception as e:
                print('*** Could not download', url)
                raise e

        ## Load a video to 'inspire' the random music

        # Loop through the chunks
        length = tfutil.get_length(arg.input) # read through the video to get the nr of frames.
        gen = skvideo.io.vreader(arg.input, num_frames=length if arg.limit is None else arg.limit)  # movie frame generator

        features = []

        print('Computing features')
        for i, frames in tqdm.tqdm(enumerate(tfutil.chunks(gen, size=FRAMECHUNK)), total=(length//FRAMECHUNK)+1):

            frames = np.concatenate([f[None, :, :, :] for f in frames], axis=0)
            # print('Loaded frame-chunk {}, with shape {}'.format(i, frames.shape))

            if shape is not None: # Resize the frame batch for the encoder model
                inshape = frames.shape

                frames = frames.transpose((1, 2, 3, 0))
                frames = frames.reshape(inshape[1], inshape[2], -1)
                frames = resize(frames, shape)
                frames = frames.reshape(shape[0], shape[1], 3, -1)
                frames = frames.transpose((3, 0, 1, 2))

            # print('   after resize:', frames.shape)
            frames = prep(frames)

            # Map to image features (1)
            features.append(encoder.predict(frames))

        features = np.concatenate(features, axis=0).squeeze()
        features = features.reshape(-1, flat)

        print('Computed features (shape {})'.format(features.shape))

        print(features[:, :10].var(axis=1))

        b, fdim = features.shape

        # Apply PCA
        pca = PCA(n_components=latent_size, whiten=True)
        z = pca.fit_transform(features)

        print(z.shape)
        print('per dimension variance (first 10)',  z[:, :10].var(axis=1))
        print('per z norm',  np.linalg.norm(z, axis=1))

        # Average over chunks of 50 frames so that each vector in the sequence
        # correponds to 2 bars
        chunks = []
        for f in range(0, b, frames_per_chunk):
            t = min(f + frames_per_chunk, b)
            chunks.append(z[f:t, :].mean(axis=0, keepdims=True))

        z = np.concatenate(chunks, axis=0)

        print('Averaged z vectors', z.shape)
        print(z[:, :10].var(axis=1))

        # Whiten (averaging will have decreasesed the variance, so we adjust the spread)
        z -= z.mean(axis=0, keepdims=True)
        z /= z.var(axis=0, keepdims=True)

        print('Whitened. per z norm', np.linalg.norm(z, axis=1))

    if arg.normalize:
        z = z / np.linalg.norm(z, axis=1, keepdims=True)

        print('Normalized. per z norm', np.linalg.norm(z, axis=1))

    z = z * arg.zmult

    # Generate MIDI (3)
    b, zdim = z.shape

    noise = np.repeat(np.random.randn(1, zdim), b, axis=0)
    z = np.concatenate([z, noise], axis=1)
    # -- We use the same epsilon noise vector throughout the video. That way, if subsequent chunks are similar, the
    #    resulting bars of music will also be similar. Resampling the noise for each chunk would lead to a change in
    #    style every 2 bars.

    ## Sample the music
    clength = arg.chunk_length * 16
    note_sequences = decoder.decode(z=z, length=clength, temperature=arg.temp)

    print(len(note_sequences), ' note sequences produced')

    # for i, ns in enumerate(note_sequences):
    #     print('chunk ', i, ns.total_time)
    #     ## Output the MIDI file
    #     mm.sequence_proto_to_midi_file(ns, '{}.{:03}.mid'.format(arg.name, i))

    # note_sequence = mm.concatenate_sequences(note_sequences, [0.75] * len(note_sequences))
    note_sequence = mm.sequences_lib.concatenate_sequences(note_sequences, [arg.chunk_length * SECONDS_PER_BAR] * len(note_sequences))

    print('total time', note_sequence.total_time)

    ## Output the MIDI file
    mm.sequence_proto_to_midi_file(note_sequence, '{}.mid'.format(arg.name))

    ## Output the sequenced MIDI as a WAV file
    # Crappy synthesizer (probably to do with missing sound font)
    # pmidi = mm.midi_io.note_sequence_to_pretty_midi(note_sequence)
    # sequenced = pmidi.synthesize(fs=SAMPLE_RATE)
    #
    # with open(arg.name + '.wav', 'wb') as file:
    #     wavfile.write(file, SAMPLE_RATE, sequenced)

    ## Use timidity to convert the MIDI to WAV

    # Run the following command: timidity output.mid -Ow
    call(['timidity', arg.name+'.mid', '-Ow'])

    ## Output the combined audio/video
    if has_video:
        # Run the following command: ffmpeg -i input.mp4 -i output.wav -c copy -map 0:v:0 -map 1:a:0 output.mkv
        call(['ffmpeg', '-i', arg.input, '-i',  arg.name+'.wav', '-c', 'copy', '-map', '0:v:0', '-map', '1:a:0', arg.name+'.mkv'])

    print('Finished')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-i", "--input",
                        dest="input",
                        help="Input movie file. Most common formats should work. The following keywords activate a special mode: 'none' generate music only, 'slerp' generate randomly interpolated music, 'random' download a random video file from the openbeelden archive.",
                        default=None, type=str)

    parser.add_argument("-m", "--mapper",
                        dest="latent-map",
                        help="Model to map the image features to the latent space. If none, PCA mapping is used.",
                        default=None, type=str)

    parser.add_argument("-t", "--temperature",
                        dest="temp",
                        help="Decoding temperature. (Higher temperature results in more variation, ",
                        default=0.5, type=float)

    parser.add_argument("-M", "--mult",
                        dest="zmult",
                        help="Multiplier for the latent vectors. Set higher than 1 to create more extreme variation.",
                        default=1.0, type=float)

    parser.add_argument("-c", "--chunk-length",
                        dest="chunk_length",
                        help="The length (in bars, lasting 2 seconds) of a chunk of frames for which the model generates a sequence of music.",
                        default=2, type=int)

    parser.add_argument("-n", "--name",
                        dest="name",
                        help="Name of the output files (without extension).",
                        default='output', type=str)

    parser.add_argument("-F", "--final", dest="final",
                        help="Use the canonical test set instead of a validation split.",
                        action="store_true")

    parser.add_argument("-N", "--normalize", dest="normalize",
                        help="Project the z vectors onto the hypersphere.",
                        action="store_true")

    parser.add_argument("--limit",
                        dest="limit",
                        help="Limit the number of frames loaded (useful for debugging).",
                        default=None, type=int)

    parser.add_argument("--encoder-model",
                        dest="encoder",
                        help="Which model to use for feature extraction (inceptionv3, mobilenetv2)",
                        default='inceptionv3', type=str)

    parser.add_argument("--decoder-model",
                        dest="decoder",
                        help="Which model to use for decoding (melody, drums, poly)",
                        default='melody', type=str)

    parser.add_argument("--model-dir",
                        dest="model_dir",
                        help="Directory to keep the downloaded models.",
                        default=None, type=str)

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)