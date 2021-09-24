import Spatial
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

IMG_PATH = "azur2021527224121.png"

img = cv.imread(IMG_PATH, 0)
gray_img = Spatial.rgb2gray(img)
kernel = np.ones((3,3))
in_pic = np.arange(25).reshape((5,5))
out = Spatial.image_convolution(in_pic, kernel)

