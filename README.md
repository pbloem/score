SCORE is a set of tools for audiovisual artists to use deep learning to generate new  
music, inspired by existing video. 

## Installation

### Python 

We use python 3. If you do not have python installed, we recommend installing the 
[Anaconda distribution](https://www.anaconda.com/download) for the most recent version
 of Python 3. 

The following command should install all required dependencies:

```
pip install keras scikit magenta wget tqdm tensorflow
```

NB: If you want to use GPU acceleration, remove tensorflow from the above command 
and [install a GPU enabled version](https://www.tensorflow.org/install/gpu). 

Additionally, if you want the script to synthesize wave files and combined video 
and audio, you'll need to install timidity and ffpeg. On macOS, install [homebrew]() and run

```brew install timidity ffmpeg```

Under Ubuntu (and other Debian-inspired distros), run

```apt-get install timidity ffmpeg```

Under Windows, you'll have to install these manually.

Once everything is installed, clone or download the project from git. 

## Usage

### Generating music

Move to the directory where you put the project files. The script ```generate.py``` allows you 
to generate music. To test if everything works, run:
 
 ``` python generate.py -i none```
 
You will find the output under ```output.mid``` and ```output.wav```. 

This generates 12 bars of monophonic music (with each 2-bar chunk unrelated to the others). To 
generate drums, use

 ``` python generate.py -i none --decoder-model drums```

And to generate polyphonic music (melody, drums and bass) use
 
``` python generate.py -i none --decoder-model poly```

The script can also generate an _interpolation_. That is, it draws a curve in latent space, and 
generates 2-bar chunks along this line. Practically, this means you will get a sequence of music 
that slowly changes from one style to another. 

``` python generate.py -i slerp```

### Video input

To generate music for a video, pass the video file as input:

```python generate.py -i video.mp4```

Most video formats should work. The file ```openbeelden.clean.csv``` contains URLs for about 3000 
videos from the Sound and Vision openbeelden archive, which should all work.

The script contains two encoder models: ```inceptionv3``` (big and slow, but close to state of the 
art) and ```mobilenetv2``` (smaller and faster). Inception is the default, you can choose moblenet as follows:
 
```python generate.py -i video.mp4 --encoder-model mobilenetv2```

To see more options, use 

```python generate.py -h```


## Other information

[SCORE!]() is a research project funded by NWO through the KIEM call.
 
SCORE is proof-of-concept research code. We provide no guarantees, but feel free 
to create a Github issue if anything doesn't work as it should, or to 
[get in touch](mailto:score@peterbloem.nl) for any reason.  
