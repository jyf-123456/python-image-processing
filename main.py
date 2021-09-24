import Spatial
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

IMG_PATH = "azur2021527224121.png"

img = cv.imread(IMG_PATH, 0)
gray_img = Spatial.rgb2gray(img)
match = np.zeros(256)
match[0:100] = 1 / 100
out = Spatial.histogram_matching(gray_img, match)
histogram = Spatial.get_histogram(out)
x = np.arange(histogram.size)
plt.title('1')
plt.plot(x, histogram)
plt.show()
cv.imshow('img', out)
cv.waitKey(0)

