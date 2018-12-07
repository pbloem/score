SCORE is a set of tools for audiovisual artists to use deep learning to generate new  
music, inspired by existing video. 

## Installation

### Python 

We use python 3. If you do not have an environment, we recommend installing the 
Anaconda distribution for the most recent version of Python 3. 

The following command should install all required dependencies:

```
pip install keras scikit magenta wget tqdm 
```

Additionally, if you want the script to synthesize wave files and combined video 
and audio, you'll need to install timidity and ffpeg. On macOS, install [homebrew]() and run

```brew install timidity ffmpeg```

Under Ubuntu (and other Debian-inspired distros), run

```apt-get install timidity ffmpeg```

Under Windows, you'll have to install these manually.

Once everything is installed, clone or download the project from git. 

## Usage

To generate music for a video, na

## Other information

[SCORE!]() is a research project funded by NWO through the KIEM call.
 
SCORE is proof-of-concept research code. We provide no guarantees, but feel free 
to create a Github issue if anything doesn't work as it should, or to 
[get in touch](mailto:score@peterbloem.nl) for any reason.  
