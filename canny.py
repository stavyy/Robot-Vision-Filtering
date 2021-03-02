# Stavros Avdella
# 3939968
# Robot Vision Spring 2019
# Programming Assignment 1
# Problem Canny

import cv2
import numpy as np
from matplotlib import pyplot as plt


# Read images
imga = cv2.imread('canny1.jpg')
imgb = cv2.imread('canny2.jpg')

img = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)

# Output data type to be able to detect edges better.
ddepth = cv2.CV_16S

# Create the kernel by creating a an 1D-array filled with the values for Fast Gaussian Smoothing,
# and multiplying by 3 we are doing this by filtering the image in one direction first (x),
# and then the other direction (y)
kernel_gs3x   = np.array(3*  [1/16, 1/8, 1/16])
kernel_gs3y   = np.array(3*  [[1/16], [1/8], [1/16]])

# Using the filter2D function to convolve the Gaussian Smoothing kernels with canny1 and canny2
a  = cv2.filter2D(img,-1,kernel_gs3x)
dst  = cv2.filter2D(a,-1,kernel_gs3y)

b = cv2.filter2D(img2,-1,kernel_gs3x)
dst2 = cv2.filter2D(b,-1,kernel_gs3y)


# Plotting  images
# 1
plt.subplot(6, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original 1')
plt.xticks([])
plt.yticks([])

# 2
plt.subplot(6, 2, 2)
plt.imshow(img2, cmap='gray')
plt.title('Original 2')
plt.xticks([])
plt.yticks([])

# 3
plt.subplot(6, 2, 3)
plt.imshow(dst, cmap='gray')
plt.title('Gaussian 1')
plt.xticks([])
plt.yticks([])

# 4
plt.subplot(6, 2, 4)
plt.imshow(dst2, cmap='gray')
plt.title('Gaussian 2')
plt.xticks([])
plt.yticks([])


# Creating kernels for forward, backward and central
kernel_f = np.array([-1,1])
kernel_b = np.array([[1,-1]])
kernel_c = np.array([[-1, 0, 1]])

# Using the filter2D function to convolve the kernels the image for each kernel
grad01 = cv2.filter2D(dst,-1, kernel_f)
grad02 = cv2.filter2D(dst,-1,kernel_b)
grad03 = cv2.filter2D(dst,-1,kernel_c)

grad04 = cv2.filter2D(dst2,-1, kernel_f)
grad05 = cv2.filter2D(dst2,-1,kernel_b)
grad06 = cv2.filter2D(dst2,-1,kernel_c)

# 1
plt.subplot(6, 2, 5)
plt.imshow(grad01, cmap='gray')
plt.title('Canny 1 Forward')
plt.xticks([])
plt.yticks([])

# 2
plt.subplot(6, 2, 7)
plt.imshow(grad02, cmap='gray')
plt.title('Canny 1 Backward')
plt.xticks([])
plt.yticks([])

# 3
plt.subplot(6, 2, 9)
plt.imshow(grad03, cmap='gray')
plt.title('Canny 1 Central')
plt.xticks([])
plt.yticks([])

# 4
plt.subplot(6, 2, 6)
plt.imshow(grad04, cmap='gray')
plt.title('Canny 2 Forward')
plt.xticks([])
plt.yticks([])

# 5
plt.subplot(6, 2, 8)
plt.imshow(grad05, cmap='gray')
plt.title('Canny 2 Backward')
plt.xticks([])
plt.yticks([])

# 6
plt.subplot(6, 2, 10)
plt.imshow(grad06, cmap='gray')
plt.title('Canny 2 Central ')
plt.xticks([])
plt.yticks([])


# Create the kernel by creating a an 2D-array filled with the values for a vertical and horizontal Sobel filter.
# We can use .T to transpose the first array to save space in our code.
kernel_h = np.array([[ 1, 0, -1],
                     [ 2, 0, -2],
                     [ 1, 0, -1]])

kernel_v = kernel_h.T

# Using the filter2D function to convolve the horizontal and vertical kernel with image 1 and 2
gv = cv2.filter2D(img, ddepth, kernel_h)
gh = cv2.filter2D(img, ddepth, kernel_v)

gv2 = cv2.filter2D(img2, -2, kernel_h)
gh2 = cv2.filter2D(img2, -2, kernel_v)

# Using the convertScale function we Scales, then calculate the absolute values, and converts the result to 8-bit.
# Using the addWeighted function we calculated the weighted sum of the two arrays.
gvabs = cv2.convertScaleAbs(gv)
ghabs = cv2.convertScaleAbs(gh)
grad = cv2.addWeighted(gvabs, 0.5, ghabs, 0.5, 0)

gvabs = cv2.convertScaleAbs(gv2)
ghabs = cv2.convertScaleAbs(gh2)
grad2 = cv2.addWeighted(gvabs, 0.5, ghabs, 0.5, 0)


# Plotting the images in a 3x2 grid system with no x and y markers

# 2
plt.subplot(6,2,11)
plt.imshow(grad, cmap='gray')
plt.title('Canny Image 1' )
plt.xticks([])
plt.yticks([])

# 3
plt.subplot(6,2,12)
plt.imshow(grad2 , cmap='gray')
plt.title('Canny Image 2')
plt.xticks([])
plt.yticks([])


# Show the results
plt.show()

# Results:
