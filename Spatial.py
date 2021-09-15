"""
    Functions from this file are all about image processing in spatial domain.
"""

import numpy as np


class Spatial:

    def __init__(self, img):
        self.__origin_img__ = img
        self.__gray_img = self.__rgb2gray()
        self.__histogram = self.__get_histogram()

        # in_pic is in rgb domain, output is in spatial domain.

    def __rgb2gray(self):
        in_pic = self.__origin_img__
        if in_pic.ndim == 2:
            out = in_pic
        else:
            out = in_pic[:, :, 0] * 0.114 + in_pic[:, :, 1] * 0.587 + in_pic[:, :, 2] * 0.229
            out = out.astype(np.uint8)
        return out

    def __get_histogram(self):
        in_pic = self.__gray_img
        histogram = np.zeros(256)
        for i in in_pic.flat:
            histogram[i] += 1
        return histogram

    def gray(self):
        return self.__gray_img

    def histogram(self):
        return self.__histogram

    # get the negative of the in_pic.
    def spatial_reverse(self):
        in_pic = self.__origin_img__
        out = 255 - in_pic
        return out

    def log_transform(self):
        in_pic = self.__origin_img__
        from math import log
        c = 255 / log(256)
        # use 1e-5 to avoid warning.
        out = c * np.log(1 + in_pic + 1e-5)
        out = out.astype(np.uint8)
        return out

    def gamma_transform(self, gamma):
        in_pic = self.__origin_img__
        gamma = float(gamma)
        c = 255 / pow(255, gamma)
        out = c * np.power(in_pic, gamma)
        out = out.astype(np.uint8)
        return out


