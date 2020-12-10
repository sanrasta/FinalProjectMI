import cv2
import numpy as np

import numpy.random as rand
import math

def loadImage(image_path):
    print(image_path)
    image = cv2.imread(image_path, 0)
    return image


def loadMatrix(filename):
    matrix = None

    return matrix


def saveImage(filename, image):

    return True


def saveMatrix(filename, matrix):

    return True


# map input image to values from 0 to 255"
def normalizeImage(image):
    normalized = None
    return normalized


# Remember: the DFT its a decomposition of signals
#  To be able to save it as an image you must convert it.
def writableDFT(dft_image):
    converted = None
    return converted


# Use openCV to display your image"
# Remember: normalize binary masks and convert FFT matrices to be able to see and save them"
def displayImage(image):
    cv2.namedWindow("Image")
    cv2.imshow("Image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def getDFT(image):
    newMatrix = None


    return None


# Convert from fft matrix to an image"
def getImage(dft_img):
    return None


# Both input values must be raw values"
def applyMask(image_dft, mask):
    return image_dft * mask


def signalToNoise():
    return False


# [Provide] Use this function to acomplish a good final image
def post_process_image(image):
    a = np.min(image)
    b = np.max(image)
    k = 255
    image = (image - a) * (k / (b - a))
    return image.astype('uint8')
