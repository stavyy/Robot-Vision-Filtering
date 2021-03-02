# Stavros Avdella
# 3939968
# Robot Vision Spring 2019
# Programming Assignment 1
# Problem 1

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read images
img = cv2.imread('image1.png')
img2 = cv2.imread('image2.png')

# Image 1 & 2 Kernel
# Create the kernel by creating a (x by x) 2D Array filled with ones and taking the average.
kernel_1 = np.ones((3,3),np.float32)/9
kernel_2 = np.ones((5,5),np.float32)/25

# Using the filter2D function to convolve the kernel with image 1
dst = cv2.filter2D(img,-1,kernel_1)
dst2 = cv2.filter2D(img,-1,kernel_2)

# Using the filter2D function to convolve the kernel with image 2
dst3 = cv2.filter2D(img2,-1,kernel_1)
dst4 = cv2.filter2D(img2,-1,kernel_2)

# Plotting the images in a 2x2 grid system with no x and y markers
# 1
plt.subplot(2,2,1)
plt.imshow(dst)
plt.title('Image 1 (3x3)')
plt.xticks([])
plt.yticks([])

# 2
plt.subplot(2,2,3)
plt.imshow(dst2)
plt.title('Image 1 (5x5)')
plt.xticks([])
plt.yticks([])

# 3
plt.subplot(2,2,2)
plt.imshow(dst3)
plt.title('Image 2 (3x3)')
plt.xticks([])
plt.yticks([])

# 4
plt.subplot(2,2,4)
plt.imshow(dst4)
plt.title('Image 2 (5x5)')
plt.xticks([])
plt.yticks([])

# Show the results
plt.show()

# Results:
# As shown after running the program the box filtering method that replaces each pixel with
# an average of its neighborhood pixels we get an output image that is cleaner and smoother.
# Image 1 is a good example in showing how box filters can remove noise from an image.
# Image 2 shows how it can give a smoother transition in colors by looking at the left
# side of the images as it goes from black to white, it makes a smooth transition rather
# than a boxy transition.