# Stavros Avdella
# 3939968
# Robot Vision Spring 2019
# Programming Assignment 1
# Problem 2

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# A median filter cannot be expressed as a convolution so we cannot create a kernel in this case since we have to
# take the image and convert it in the form of an array. From there depending on the size of our filter we take the
# elements in that filter size and take the median. That will be the new value for our filtered image pixel.
def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final


def main():
    img = Image.open("image1.png").convert("L")
    img2 = Image.open("image2.png").convert("L")
    arr = np.array(img)
    arr2 = np.array(img2)

    removed_noisei1 = median_filter(arr, 1)
    removed_noisei2 = median_filter(arr2, 1)

    removed_noise  = median_filter(arr, 3)
    removed_noise2 = median_filter(arr, 5)
    removed_noise3 = median_filter(arr, 7)
    removed_noise4 = median_filter(arr2, 3)
    removed_noise5 = median_filter(arr2, 5)
    removed_noise6 = median_filter(arr2, 9)

    imgi1 = Image.fromarray(removed_noisei1)
    imgi2 = Image.fromarray(removed_noisei2)

    img  = Image.fromarray(removed_noise)
    img2 = Image.fromarray(removed_noise2)
    img3 = Image.fromarray(removed_noise3)
    img4 = Image.fromarray(removed_noise4)
    img5 = Image.fromarray(removed_noise5)
    img6 = Image.fromarray(removed_noise6)

    # 2
    plt.subplot(4, 2, 1)
    plt.imshow(imgi1)
    plt.title('Image 1 Original')
    plt.xticks([])
    plt.yticks([])

    # 2
    plt.subplot(4, 2, 2)
    plt.imshow(imgi2)
    plt.title('Image 2 Original')
    plt.xticks([])
    plt.yticks([])

    # 2
    plt.subplot(4,2,3)
    plt.imshow(img)
    plt.title('Image 1 (3x3)')
    plt.xticks([])
    plt.yticks([])

    # 2
    plt.subplot(4, 2, 5)
    plt.imshow(img2)
    plt.title('Image 1 (5x5)')
    plt.xticks([])
    plt.yticks([])

    # 2
    plt.subplot(4, 2, 7)
    plt.imshow(img3)
    plt.title('Image 1 (7x7)')
    plt.xticks([])
    plt.yticks([])

    # 2
    plt.subplot(4, 2, 4)
    plt.imshow(img4)
    plt.title('Image 2 (3x3)')
    plt.xticks([])
    plt.yticks([])

    # 2
    plt.subplot(4, 2, 6)
    plt.imshow(img5)
    plt.title('Image 2 (5x5)')
    plt.xticks([])
    plt.yticks([])

    # 2
    plt.subplot(4, 2, 8)
    plt.imshow(img6)
    plt.title('Image 2 (7x7)')
    plt.xticks([])
    plt.yticks([])

    plt.show()


main()

