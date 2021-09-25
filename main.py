import Spatial
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

IMG_PATH = "azur2021527224121.png"

img = cv.imread(IMG_PATH, 0)
gray_img = Spatial.rgb2gray(img)
out = Spatial.sharpening(gray_img, method='blur', blur_method='box')
cv.imshow('img', out)
cv.waitKey(0)

