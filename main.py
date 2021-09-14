import Spatial
import cv2 as cv
import numpy as np

IMG_PATH = "azur2021527224121.png"

img = cv.imread(IMG_PATH, 0)
out = Spatial.gamma_transform(img, 2)
cv.imshow('img', out)
cv.waitKey(0)

