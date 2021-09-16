import Spatial
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

IMG_PATH = "azur2021527224121.png"

img = cv.imread(IMG_PATH, 0)
gray_img = Spatial.rgb2gray(img)
histogram = Spatial.get_histogram(gray_img)
Spatial.plot_histogram(histogram)
# cv.imshow('img', out.gray())
# cv.waitKey(0)

