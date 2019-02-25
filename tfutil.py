import skvideo

from itertools import chain, islice

import numpy as np
from numpy.linalg import norm

import os, pathlib

""" Script directory """
DIR = os.path.dirname(os.path.realpath(__file__))

def ensure(dir):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

def chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))

def get_length(file):
    gen = skvideo.io.vreader(file)

    length = 0
    for _ in gen:
        length += 1
    return length


def linp(x, y, steps=5):
    """
    Produces a spherical linear interpolation between two points

    :param x:
    :param y:
    :param steps:
    :return:
    """
    assert x.shape[0] == y.shape[0]
    n = x.shape[0]

    d = np.linspace(0, 1, steps)

    return   x[None, :] * d[:, None] \
           + y[None, :] * (1-d)[:, None]

def slerp(x, y, steps=5):
    """
    Produces a spherical linear interpolation between two points

    :param x: 1 by n matrix or length-n vector
    :param y:
    :param steps:
    :return:
    """
    assert x.shape[0] == y.shape[0]

    if len(x.shape) > 1:
        x, y = x[0], y[0]

    n = x.shape[0]

    angle = np.arccos( np.dot(x, y) / (norm(x) * norm(y)) )

    d = np.linspace(0, 1, steps)

    d1 = np.sin((1-d) * angle) / np.sin(angle)
    d2 = np.sin(d     * angle) / np.sin(angle)

    return   x[None, :] * d1[:, None] \
           + y[None, :] * d2[:, None]




if __name__ == "__main__":

    print( linp(np.ones(1), -np.ones(1)))
    print(slerp(np.ones(1), -np.ones(1)))

