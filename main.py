from Spatial import Spatial
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

IMG_PATH = "azur2021527224121.png"

img = cv.imread(IMG_PATH, 0)
out = Spatial(img)
histogram = out.histogram()
x = np.arange(256)
plt.title("1")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, histogram)
plt.show()
# cv.imshow('img', out.gray())
# cv.waitKey(0)

