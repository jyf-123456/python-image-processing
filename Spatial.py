"""
    Functions from this file are all about image processing in spatial domain.
    Only apply on 8-bit grayscale.
"""

import numpy as np


def rgb2gray(in_pic):
    if in_pic.ndim == 2:
        out = in_pic.copy()
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


# map [in_low, in_high] into [out_low, out_high]
def contrast_stretching(in_pic, in_low, in_high, out_low, out_high):
    slope_1 = out_low / in_low
    slope_2 = (out_high - out_low) / (in_high - in_low)
    slope_3 = (255 - out_high) / (255 - in_high)
    pic_shape = in_pic.shape
    out = np.zeros(pic_shape)
    for i in range(pic_shape[0]):
        for j in range(pic_shape[1]):
            gray_in = in_pic[i][j]
            gray_out = 0
            if (gray_in < in_low) and (gray_in > 0):
                gray_out = round(slope_1 * gray_in)
            elif (gray_in < in_high) and (gray_in > in_low):
                gray_out = round(out_low + slope_2 * (gray_in - in_low))
            elif (gray_in < 255) and (gray_in > in_high):
                gray_out = round(out_high + slope_3 * (gray_in - in_high))
            out[i][j] = gray_out
    out = out.astype(np.uint8)
    return out


# map [in_low, in_high} into an constant gray value.
def gray_level_slicing(in_pic, in_low, in_high, constant):
    out = in_pic.copy()
    for i in np.nditer(out, op_flags=['readwrite']):
        if (i > in_low) and (i < in_high):
            i[...] = constant
    return out


# get one bit plane from 0~7 plane.
def bit_plane_slicing(in_pic, layer):
    out = in_pic.copy()
    for i in np.nditer(out, op_flags=['readwrite']):
        # bin_value is 8-bit width.
        bin_value = bin(i)[2:].zfill(8)
        layer_bit = bin_value[-(int(layer)+1)]
        i[...] = int(layer_bit) * 255
    return out


# get the cumulative distribution function(CDF) of in_pic.
def cdf(in_pic_histogram):
    scale = in_pic_histogram.size
    transform = np.zeros(scale, dtype=np.uint8)
    temp = 0
    for i in range(scale):
        temp = temp + in_pic_histogram[i]
        transform[i] = round((scale - 1) * temp)
    return transform


def histogram_equalize(in_pic):
    out = in_pic.copy()
    in_pic_histogram = get_histogram(in_pic)
    transform = cdf(in_pic_histogram)
    for i in np.nditer(out, op_flags=['readwrite']):
        i[...] = transform[i]
    return out


# variable match is a given histogram for histogram matching.
def histogram_matching(in_pic, match):
    out = in_pic.copy()
    in_pic_histogram = get_histogram(in_pic)
    in_transform = cdf(in_pic_histogram)
    match_transform = cdf(match)
    match_transform_inverse = np.zeros(match_transform.size, dtype=np.uint8)
    for i in range(match_transform.size):
        j = match_transform[-i-1]
        match_transform_inverse[j] = i
    for i in range(match_transform_inverse.size):
        if i == 0:
            pass
        else:
            if match_transform_inverse[i] == 0:
                match_transform_inverse[i] = match_transform_inverse[i-1]
    for i in np.nditer(out, op_flags=['readwrite']):
        i[...] = in_transform[i]
        i[...] = match_transform_inverse[i]
    return out
