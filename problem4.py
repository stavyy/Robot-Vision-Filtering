# Stavros Avdella
# 3939968
# Robot Vision Spring 2019
# Programming Assignment 1
# Problem 4

import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import ndimage

# Read images and convert them to RGB. The Image is already in RGB format but when it
img = cv2.imread('image3.png')
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Creating kernels for forward, backward and central
kernel_f = np.array([-1,1])
kernel_b = np.array([[1,-1]])
kernel_c = np.array([[-1, 0, 1]])

# Using the filter2D function to convolve the kernels the image for each kernel
dst = cv2.filter2D(img,-1, kernel_f)
dst2 = cv2.filter2D(img,-1,kernel_b)
dst3 = cv2.filter2D(img,-1,kernel_c)

# Converting the image to grayscale
dst  = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2GRAY)
dst3 = cv2.cvtColor(dst3, cv2.COLOR_BGR2GRAY)

# Plotting pictures
# 1
plt.subplot(2,2,1)
plt.imshow(RGB_img)
plt.title('Original Image')
plt.xticks([])
plt.yticks([])

# 2
plt.subplot(2,2,3)
plt.imshow(dst, cmap='gray')
plt.title('Gradient forward')
plt.xticks([])
plt.yticks([])

# 3
plt.subplot(2,2,4)
plt.imshow(dst2, cmap='gray')
plt.title('Gradient backward')
plt.xticks([])
plt.yticks([])

# 4
plt.subplot(2,2,2)
plt.imshow(dst3, cmap='gray')
plt.title('Gradient central')
plt.xticks([])
plt.yticks([])

plt.show()