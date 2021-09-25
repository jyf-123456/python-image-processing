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
        layer_bit = bin_value[-(int(layer) + 1)]
        i[...] = int(layer_bit) * 255

    return out


# get the cumulative distribution function(CDF) of in_pic.
def cdf(in_pic_histogram):
    scale = in_pic_histogram.size
    transform = np.zeros(scale, dtype=np.uint8)
    temp = 0
    for i in range(scale):
        temp = temp + in_pic_histogram[i]
        transform[i] = (scale - 1) * temp

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
        j = match_transform[-i - 1]
        match_transform_inverse[j] = 255 - i

    for i in range(match_transform_inverse.size):
        if i == 0:
            pass

        else:
            if match_transform_inverse[i] == 0:
                match_transform_inverse[i] = match_transform_inverse[i - 1]

    for i in np.nditer(out, op_flags=['readwrite']):
        i[...] = in_transform[i]
        i[...] = match_transform_inverse[i]

    return out


# default padding of edge is 0.
# operate_type is 'linear', means convolution.
# when operate_type is 'order_statistic', just require kernel's shape.
def image_kernel_operation(in_pic, operate_type='linear', kernel=None):
    in_pic_shape = in_pic.shape
    out = np.zeros((in_pic_shape[0] + kernel.shape[0] - 1, in_pic_shape[1] + kernel.shape[1] - 1))
    index = kernel.shape[0] // 2, kernel.shape[1] // 2
    out[index[0]:index[0] + in_pic_shape[0], index[1]:index[1] + in_pic_shape[1]] = in_pic
    out_copy = out.copy()
    if operate_type == 'order_statistic':
        for i in range(index[0], in_pic_shape[0] + index[0]):
            for j in range(index[1], in_pic_shape[1] + index[1]):
                temp = out_copy[i - index[0]:i + index[0], j - index[1]:j + index[1]]
                temp_median = np.median(temp)
                out[i][j] = temp_median.astype(np.uint8)

    else:
        for i in range(index[0], in_pic_shape[0] + index[0]):
            for j in range(index[1], in_pic_shape[1] + index[1]):
                convolution_sum = 0

                for s in range(-index[0], index[0] + 1):
                    for t in range(-index[1], index[1] + 1):
                        convolution_sum += out_copy[i - s][j - t] * kernel[index[0] + s][index[1] + t]

                out[i][j] = round(convolution_sum)

    if np.min(out) < 0:
        out = out - np.min(out)
        for i in np.nditer(out, op_flags=['readwrite']):
            if i > 255:
                i[...] = 255

    out = out.astype(np.uint8)
    return out[index[0]:index[0] + in_pic_shape[0], index[1]:index[1] + in_pic_shape[1]]


# generate gaussian kernel, it's always a square.
def gauss_kernel(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    if sigma <= 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = sigma ** 2
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center

            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / 2 * s)
            sum_val += kernel[i, j]

    kernel = kernel / sum_val

    return kernel


# allow box filtering and gaussian filtering.
# default box filtering.
def smoothing_filter(in_pic, kernel_shape, kernel_type='box', sigma=1):
    if kernel_type == 'gaussian':
        try:
            if kernel_shape[0] != kernel_shape[1]:
                raise Exception("given kernel shape is not a square.")

        except Exception as e:
            print(e)
            return None

        else:
            kernel = gauss_kernel(kernel_shape[0], sigma)

        out = image_kernel_operation(in_pic, operate_type='linear', kernel=kernel)

    elif kernel_type == 'order_statistic':
        kernel = np.zeros(shape=kernel_shape)
        out = image_kernel_operation(in_pic, operate_type='order_statistic', kernel=kernel)

    else:
        kernel = np.ones(shape=kernel_shape) / (kernel_shape[0] * kernel_shape[1])
        out = image_kernel_operation(in_pic, operate_type='linear', kernel=kernel)

    return out


def sharpening(in_pic, method='laplacian', blur_method='box', kernel_shape=(3, 3), sigma=1):
    if method == 'blur':
        if blur_method == 'gaussian':
            sharpen_model = in_pic - smoothing_filter(in_pic, kernel_shape, kernel_type='gaussian', sigma=sigma)

        elif blur_method == 'order_statistic':
            sharpen_model = in_pic - smoothing_filter(in_pic, kernel_shape, kernel_type='order_statistic')

        else:
            sharpen_model = in_pic - smoothing_filter(in_pic, kernel_shape, kernel_type='box')

    else:
        laplacian = np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        sharpen_model = image_kernel_operation(in_pic, operate_type='linear', kernel=laplacian)

    out = sharpen_model
    return out
