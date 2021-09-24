import Spatial
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

IMG_PATH = "azur2021527224121.png"

img = cv.imread(IMG_PATH, 0)
gray_img = Spatial.rgb2gray(img)
transform = Spatial.cdf(gray_img)
x = np.arange(transform.size)
plt.title('1')
plt.plot(x, transform)
plt.show()
# cv.imshow('img', out)
# cv.waitKey(0)

