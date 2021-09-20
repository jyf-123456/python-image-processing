import Spatial
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

IMG_PATH = "azur2021527224121.png"

img = cv.imread(IMG_PATH, 0)
gray_img = Spatial.rgb2gray(img)
out = Spatial.histogram_equalize(gray_img)
in_histogram = Spatial.get_histogram(gray_img)
out_histogram = Spatial.get_histogram(out)
Spatial.plot_histogram(in_histogram, title='in')
Spatial.plot_histogram(out_histogram, title='out')
cv.imshow('img', out)
cv.waitKey(0)

