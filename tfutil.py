import skvideo

from itertools import chain, islice

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


if __name__ == "__main__":

    for c in chunks(range(100)):
        print(list(c))