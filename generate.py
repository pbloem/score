from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

from argparse import ArgumentParser

"""
Generate a MIDI track from a movie file.

The pipeline used by this script works as follows.

1) A pretrained model (InceptionV3) is used to map the each frame in the video to a vector of _image features_.
2) _mapper_ is used to convert each cluster of n frames to a latent vectors representing 16 bars of music. The mapper
ensures that the resulting latent variables fit the standard normal shape of the latent space. Two mappers are available:
 * PCA: This mapper only looks at the frames of the current video, and fits them to a standard normal distribution
using a Principal Component Analysis. This is the default mapper.
 * vae.???: This mapper is the encoder part of a VAE trained on image features extracted from the the whole openbeelden archive.
 Choose this mapper by adding the argument the argument "-m vae.??".
3) The resulting latent vectors are fed to the MusicVAE decoder to generate 16 bars of music.

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
* Any tensorflow model that maps a (batch, ???)-tensor to a (batch, ???) tensor can be used as a custom mapper. Just
save the model and load it with the "-m" switch. For good results, the model should map to points that are likely under
the multivariate standard normal distribution.

"""


def go(arg):
    # Load movie frames


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
                        default=0.001, type=float)

    parser.add_argument("--do",
                        dest="do",
                        help="Dropout",
                        default=None, type=float)

    parser.add_argument("-D", "--dataset-name",
                        dest="name",
                        help="Name of dataset to use [aifb, am]",
                        default='aifb', type=str)

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
                        help="Limit the number of relations.",
                        default=None, type=int)

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)