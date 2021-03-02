# Stavros Avdella
# 3939968
# Robot Vision Spring 2019
# Programming Assignment 1
# Problem 7

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
img = cv2.imread('image4.png')


# Displaying a histogram with bin size of 256
hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.hist(img.flatten(),256,[0,256], color = 'g')
plt.xlim([0,256])
plt.show()


# Displaying a histogram with bin size of 128
hist,bins = np.histogram(img.flatten(),128,[0,128])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.hist(img.flatten(),128,[0,128], color = 'b')
plt.xlim([0,128])
plt.show()


# Displaying a histogram with bin size of 64
hist,bins = np.histogram(img.flatten(),64,[0,64])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.hist(img.flatten(),64,[0,64], color = 'b')
plt.xlim([0,64])
plt.show()