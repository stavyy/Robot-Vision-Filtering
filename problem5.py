# Stavros Avdella
# 3939968
# Robot Vision Spring 2019
# Programming Assignment 1
# Problem 5

import cv2
import numpy as np
from matplotlib import pyplot as plt


# Read images
img = cv2.imread('image1.png')
img2 = cv2.imread('image2.png')

# Output data type to be able to detect edges better.
ddepth = cv2.CV_16S

# Create the kernel by creating a an 2D-array filled with the values for a vertical and horizontal Sobel filter.
# We can use .T to transpose the first array to save space in our code.
kernel_h = np.array([[ 1, 0, -1],
                     [ 2, 0, -2],
                     [ 1, 0, -1]])

kernel_v = kernel_h.T

# Using the filter2D function to convolve the horizontal and vertical kernel with image 1 and 2
gv = cv2.filter2D(img, ddepth, kernel_h)
gh = cv2.filter2D(img, ddepth, kernel_v)

gv2 = cv2.filter2D(img2, ddepth, kernel_h)
gh2 = cv2.filter2D(img2, ddepth, kernel_v)

# Using the convertScale function we Scales, then calculate the absolute values, and converts the result to 8-bit.
# Using the addWeighted function we calculated the weighted sum of the two arrays.
gvabs = cv2.convertScaleAbs(gv)
ghabs = cv2.convertScaleAbs(gh)
grad = cv2.addWeighted(gvabs, 0.5, ghabs, 0.5, 0)

gvabs = cv2.convertScaleAbs(gv2)
ghabs = cv2.convertScaleAbs(gh2)
grad2 = cv2.addWeighted(gvabs, 0.5, ghabs, 0.5, 0)

# Plotting the images in a 3x2 grid system with no x and y markers
# 1
plt.subplot(2,2,1)
plt.imshow(img)
plt.title('Original Image 1')
plt.xticks([])
plt.yticks([])

# 2
plt.subplot(2,2,3)
plt.imshow(grad)
plt.title('Sobel Image 1')
plt.xticks([])
plt.yticks([])

# 3
plt.subplot(2,2,4)
plt.imshow(grad2)
plt.title('Sobel Image 2')
plt.xticks([])
plt.yticks([])

# 4
plt.subplot(2,2,2)
plt.imshow(img2)
plt.title('Original Image 2')
plt.xticks([])
plt.yticks([])

# Show the results
plt.show()

# Results:
# They are a bit messy due to images being used.
# The program can detect most horizontal and vertical edges.
# However, as you can see it is able to detect edges better
# in image 2 because there is much less noise in that image.