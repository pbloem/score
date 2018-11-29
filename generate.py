from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import numpy as np

import magenta.music as mm
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel

import skvideo.io
import wget, tqdm, os, tfutil, sys

from scipy.io import wavfile

from argparse import ArgumentParser

from subprocess import call

# suppress warnings
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from sklearn.decomposition import PCA

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
* The generated music always consists of ...-second bars in 4/4 time, with a melody, bass, and drum line. (This is a
limitation of the MusicVAE model).
* (TODO) Any tensorflow model that maps a (b*???, ???)-tensor to a (b, 512) tensor can be used as a custom mapper. Just
save the model and load it with the "-m" switch. For good results, the model should map to points that are likely under
the multivariate standard normal distribution.
"""

FPS = 25             # We assume PAL for now
BARS_PER_CHUNK = 2   # Use the 2-bar model
SECONDS_PER_BAR = 2  # Default MIDI timing (120BPM, 480 BPQ)
MVAE_URL = 'https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/cat-mel_2bar_big.tar'
SAMPLE_RATE = 44100
FRAMECHUNK = 10000 # Set as big as memory allows
LATENTSIZE = 512

def go(arg):

    # Load pretrained models

    ## InceptionV3 (to map frames to image features)
    encoder = InceptionV3(weights='imagenet', include_top=False)
    frames_per_chunk = BARS_PER_CHUNK * SECONDS_PER_BAR * FPS

    if arg.input == 'random':
        ## Generate 6 bars of random music
        z = np.random.randn(3, LATENTSIZE)

    else:
        ## Load a video to 'inspire' the random music

        # Loop through the chunks
        length = tfutil.get_length(arg.input) # read through the video to get the nr of frames.
        gen = skvideo.io.vreader(arg.input, num_frames=length if arg.limit is None else arg.limit)  # movie frame generator

        features = []

        for i, frames in enumerate(tfutil.chunks(gen, size=FRAMECHUNK)):

            frames = np.concatenate([f[None, :, :, :] for f in frames], axis=0)
            print('Loaded frame-chunk {}, with shape {}'.format(i, frames.shape))

            frames = preprocess_input(frames)

            # Map to image features (1)
            features.append(encoder.predict(frames)[None, :])

        features = np.concatenate(features, axis=0).squeeze()
        features = features.reshape(-1, 6 * 8 * 2048)

        print('Computed features (shape {})'.format(features.shape))

        print(features[:, :10].var(axis=1))

        b, fdim = features.shape

        # Average over chunks of 50 frames so that each vector in the sequence
        # correponds to 2 bars
        chunks = []
        for f in range(0, b, frames_per_chunk):

            t = min(f + frames_per_chunk, b)
            chunks.append(features[f:t, :].mean(axis=0, keepdims=True))
        features = np.concatenate(chunks, axis=0)

        print('Averages feature vectors', features.shape)
        print(features[:, :10].var(axis=1))

        # Apply PCA
        pca = PCA(n_components=LATENTSIZE, whiten=True)
        z = pca.fit_transform(features)

        print(z.shape)
        print('per dimension variance (first 10)',  z[:, :10].var(axis=1))
        print('per z norm',  np.linalg.norm(z, axis=1))

    # Generate MIDI (3)
    b, zdim = z.shape

    ## Load the Music VAE model
    mfile = 'musicmodel.tar'
    if not os.path.isfile(mfile):
        print('Downloading MusicVAE')
        wget.download(MVAE_URL, mfile)

    decoder_config = configs.CONFIG_MAP['cat-mel_2bar_big']
    decoder = TrainedModel(decoder_config, batch_size=4, checkpoint_dir_or_path='./musicmodel.tar')

    noise = np.repeat(np.random.randn(1, zdim), b, axis=0)
    # -- We use the same epsilon noise vector throughout the video. That way, if subsequent chunks are similar, the
    #    resulting bars of music will also be similar. Resampling the noise for each chunk would lead to a change in
    #    style every 2 bars.

    ## Sample the music
    note_sequences = decoder.decode(z=z, length=32, temperature=arg.temp)

    for n in note_sequences:
        print(n)
    sys.exit()

    ## Output the MIDI file
    mm.sequence_proto_to_midi_file(note_sequence, arg.name + '.mid')

    ## Output the sequenced MIDI as a WAV file
    pmidi = mm.midi_io.note_sequence_to_pretty_midi(note_sequence)
    sequenced = pmidi.synthesize(fs=SAMPLE_RATE, sf2_path=None)

    with open(arg.name + '.wav', 'wb') as file:
        wavfile.write(file, SAMPLE_RATE, sequenced)

    ## Output the combined audio/video

    # Run the following command: ffmpeg -i input.mp4 -i output.wav -c copy -map 0:v:0 -map 1:a:0 output.mkv
    call(['ffmpeg', '-i', arg.input, '-i',  arg.name+'.wav', '-c', 'copy', '-map', '0:v:0', '-map', '1:a:0', arg.name+'.mkv'])

    print('Finished')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-i", "--input",
                        dest="input",
                        help="Input movie file. Preferably a ",
                        default=None, type=str)

    parser.add_argument("-m", "--mapper",
                        dest="latent-map",
                        help="Model to map the image features to the latent space. If none, PCA mapping is used.",
                        default=None, type=str)

    parser.add_argument("-t", "--temperature",
                        dest="temp",
                        help="Decoding temperature. (Higher temperature results in more variation, ",
                        default=0.5, type=float)

    parser.add_argument("-n", "--name",
                        dest="name",
                        help="Name of the output files (without extension).",
                        default='output', type=str)

    parser.add_argument("-N", "--normalize",
                        dest="normalize",
                        help="Normalize the embeddings.",
                        action="store_true")

    parser.add_argument("-F", "--final", dest="final",
                        help="Use the canonical test set instead of a validation split.",
                        action="store_true")

    parser.add_argument("-U", "--unidir", dest="unidir",
                        help="Only model relations in one direction.",
                        action="store_true")

    parser.add_argument("--dense", dest="dense",
                        help="Use a dense adjacency matrix with the canonical softmax.",
                        action="store_true")

    parser.add_argument("--limit",
                        dest="limit",
                        help="Limit the number of frames (useful for debugging).",
                        default=None, type=int)

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)