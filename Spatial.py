"""
    Functions from this file are all about image processing in spatial domain.
"""

import numpy as np


def rgb2gray(in_pic):
    if in_pic.ndim == 2:
        out = in_pic
    else:
        out = in_pic[:, :, 0] * 0.114 + in_pic[:, :, 1] * 0.587 + in_pic[:, :, 2] * 0.229
        out = out.astype(np.uint8)
    return out


# in_pic is in gray domain.
def get_histogram(in_pic, scale=256):
    histogram = np.zeros(scale)
    pic_size = in_pic.size
    for i in in_pic.flat:
        histogram[i] += 1
    histogram /= pic_size
    return histogram


def plot_histogram(histogram, title="Histogram", xlabel="Grayscale", ylabel="Probability"):
    from matplotlib import pyplot as plt
    x = np.arange(histogram.size)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, histogram)
    plt.show()


# get the negative of the in_pic.
def spatial_reverse(in_pic):
    out = 255 - in_pic
    return out


def log_transform(in_pic):
    from math import log
    c = 255 / log(256)
    # use 1e-5 to avoid warning.
    out = c * np.log(1 + in_pic + 1e-5)
    out = out.astype(np.uint8)
    return out


def gamma_transform(in_pic, gamma):
    gamma = float(gamma)
    c = 255 / pow(255, gamma)
    out = c * np.power(in_pic, gamma)
    out = out.astype(np.uint8)
    return out


