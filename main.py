import Spatial
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

IMG_PATH = "test.jpg"

img = cv.imread(IMG_PATH, 0)
gray_img = Spatial.rgb2gray(img)
out = Spatial.sharpening(gray_img, method='blur', blur_method='box')
cv.imshow('img', gray_img)
cv.imshow('1', out)
cv.waitKey(0)

