"""
    Functions from this file are all about image processing in spatial domain.
"""

import numpy as np


# in_pic is in rgb domain, output is in spatial domain.
def rgb2gray(in_pic):
    out = in_pic[:, :, 0] * 0.114 + in_pic[:, :, 1] * 0.587 + in_pic[:, :, 2] * 0.229
    out = out.astype(np.uint8)
    return out


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
