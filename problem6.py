# Stavros Avdella
# 3939968
# Robot Vision Spring 2019
# Programming Assignment 1
# Problem 6

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read images
img = cv2.imread('image1.png')
img2 = cv2.imread('image2.png')

# Create the kernel by creating a an 1D-array filled with the values for Fast Gaussian Smoothing,
# and multiplying by 3, 5, and 10, we are doing this by filtering the image in one direction first (x),
# and then the other direction (y)
kernel_gs3x   = np.array(3*  [1/16, 1/8, 1/16])
kernel_gs3y   = np.array(3*  [[1/16], [1/8], [1/16]])

kernel_gs5x   = np.array(5*  [1/16, 1/8, 1/16])
kernel_gs5y   = np.array(5*  [[1/16], [1/8], [1/16]])

kernel_gs10x  = np.array(10* [1/16, 1/8, 1/16])
kernel_gs10y  = np.array(10* [[1/16], [1/8], [1/16]])

# Using the filter2D function to convolve the Gaussian Smoothing kernels with image 1 and 2
a  = cv2.filter2D(img,-1,kernel_gs3x)
dst  = cv2.filter2D(a,-1,kernel_gs3y)

b = cv2.filter2D(img,-1,kernel_gs5x)
dst2 = cv2.filter2D(b,-1,kernel_gs5y)

c = cv2.filter2D(img,-1,kernel_gs10x)
dst3 = cv2.filter2D(c,-1,kernel_gs10y)

d = cv2.filter2D(img2,-1,kernel_gs3x)
dst4 = cv2.filter2D(d,-1,kernel_gs3y)

e = cv2.filter2D(img2,-1,kernel_gs5x)
dst5 = cv2.filter2D(e,-1,kernel_gs5y)

f = cv2.filter2D(img2,-1,kernel_gs10x)
dst6 = cv2.filter2D(f,-1,kernel_gs10y)


# 1
plt.subplot(4,2,1)
plt.imshow(img)
plt.title('Original Image 1')
plt.xticks([])
plt.yticks([])

# 2
plt.subplot(4,2,3)
plt.imshow(dst)
plt.title('Fast Gaussian 3 Image 1')
plt.xticks([])
plt.yticks([])

# 3
plt.subplot(4,2,5)
plt.imshow(dst2)
plt.title('Fast Gaussian 5 Image 1')
plt.xticks([])
plt.yticks([])

# 4
plt.subplot(4,2,7)
plt.imshow(dst3)
plt.title('Fast Gaussian 10 Image 1')
plt.xticks([])
plt.yticks([])

# 5
plt.subplot(4,2,2)
plt.imshow(img2)
plt.title('Original Image 2')
plt.xticks([])
plt.yticks([])

# 6
plt.subplot(4,2,4)
plt.imshow(dst4)
plt.title('Fast Gaussian 3 Image 2')
plt.xticks([])
plt.yticks([])

# 7
plt.subplot(4,2,6)
plt.imshow(dst5)
plt.title('Fast Gaussian 5 Image 2')
plt.xticks([])
plt.yticks([])

# 8
plt.subplot(4,2,8)
plt.imshow(dst6)
plt.title('Fast Gaussian 10 Image 2')
plt.xticks([])
plt.yticks([])



# Show the results
plt.show()

# Results:
# The 2D Gaussian is much more cleaner and requires less run time.