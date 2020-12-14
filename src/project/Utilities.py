import cv2
import matplotlib.pyplot as plt
import numpy as np

import numpy.random as rand
import math

def loadImage(image_path):
    print(image_path)
    image = cv2.imread(image_path, 0)

    return image


def loadMatrix(filename):
    matrix = np.load(filename)
    return matrix


def saveImage(filename, image):
    return cv2.imwrite(filename, image)


def saveMatrix(filename, matrix):
    return cv2.imwrite(filename, matrix)


# map input image to values from 0 to 255"
def normalizeImage(image):
    height, width = image.shape
    normalized = np.zeros((height, width))
    normalized = cv2.normalize(image, normalized, 0, 255, cv2.NORM_MINMAX)
    return normalized


# Remember: the DFT its a decomposition of signals
#  To be able to save it as an image you must convert it.
def writableDFT(dft_image):
    converted = 20*np.log(np.abs(dft_image))
    return converted


# Use openCV to display your image"
# Remember: normalize binary masks and convert FFT matrices to be able to see and save them"
def displayImage(image):
    cv2.namedWindow("Image")
    cv2.imshow("Image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def displayImage_plt(image):
    plt.imshow(image, cmap = 'gray')
    plt.show()



def getDFT(image):
    newMatrix = np.fft.fft2(image)
    image_dft = np.fft.fftshift(newMatrix)
    return image_dft


# Convert from fft matrix to an image"
def getImage(image_dft):
    dft_unshifted = np.fft.ifftshift(image_dft)
    image_reverse_fft = np.fft.ifft2(dft_unshifted)
    img_back = np.abs(image_reverse_fft)
    return img_back


# Both input values must be raw values"
def applyMask(image_dft, mask):
    return image_dft * mask


def signalToNoise(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))


# [Provide] Use this function to acomplish a good final image
def post_process_image(image):
    a = np.min(image)
    b = np.max(image)
    k = 255
    image = (image - a) * (k / (b - a))
    return image.astype('uint8')
