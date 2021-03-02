# Stavros Avdella
# 3939968
# Robot Vision Spring 2019
# Programming Assignment 1
# Problem 3

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read images
img = cv2.imread('image1.png')
img2 = cv2.imread('image2.png')

# Create the kernel by creating a an 2D-array filled with the values for Gaussian Smoothing,
# and multiplying by 3, 5, and 10.
kernel_gs3  = np.array(3* [[ 1/16,  1/8, 1/16],
                          [   1/8,  1/4,  1/8],
                          [  1/16,  1/8, 1/16]])

kernel_gs5  = np.array(5* [[ 1/16,  1/8, 1/16],
                          [   1/8,  1/4,  1/8],
                          [  1/16,  1/8, 1/16]])

kernel_gs10 = np.array(10*[[ 1/16,  1/8, 1/16],
                          [   1/8,  1/4,  1/8],
                          [  1/16,  1/8, 1/16]])

# Using the filter2D function to convolve the Gaussian Smoothing kernels with image 1 and 2 for each kernel
dst  = cv2.filter2D(img,-1,kernel_gs3)
dst2 = cv2.filter2D(img,-1,kernel_gs5)
dst3 = cv2.filter2D(img,-1,kernel_gs10)

dst4 = cv2.filter2D(img2,-1,kernel_gs3)
dst5 = cv2.filter2D(img2,-1,kernel_gs5)
dst6 = cv2.filter2D(img2,-1,kernel_gs10)


# Plotting the images in a 2x4 grid system with no x and y markers
# 1
plt.subplot(4,2,1)
plt.imshow(img)
plt.title('Original Image 1')
plt.xticks([])
plt.yticks([])

# 2
plt.subplot(4,2,3)
plt.imshow(dst)
plt.title('Gaussian 3 Image 1')
plt.xticks([])
plt.yticks([])

# 3
plt.subplot(4,2,5)
plt.imshow(dst2)
plt.title('Gaussian 5 Image 1')
plt.xticks([])
plt.yticks([])

# 4
plt.subplot(4,2,7)
plt.imshow(dst3)
plt.title('Gaussian 10 Image 1')
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
plt.title('Gaussian 3 Image 2')
plt.xticks([])
plt.yticks([])

# 7
plt.subplot(4,2,6)
plt.imshow(dst5)
plt.title('Gaussian 5 Image 2')
plt.xticks([])
plt.yticks([])

# 8
plt.subplot(4,2,8)
plt.imshow(dst6)
plt.title('Gaussian 10 Image 2')
plt.xticks([])
plt.yticks([])



# Show the results
plt.show()

# Results:
# Different sigmas change the intensity of the image making the image bring out more the other lighter colors as
# you increase the sigma. Between question 1, 2, and 3. I think that median filters provide the smoothest filter
# to an image.  More specifically a 7 by 7 median filter.