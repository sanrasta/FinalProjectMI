import numpy as np
import cv2
import math

##
# Objective: Simulate different types of acquisition patterns by implementing the
# following functions.
# You can indeed use opencv functions for the ellipse, radial and spiral pattern
### also for generating GUIs and Presenting the output of your work,(displaying)
# convolution function to display (multiply convolute telnet
#inverse random transform, random transform
def cartesianPattern(mask_size, percent):
    mask = np.zeros((mask_size[0], mask_size[1]))
    rows = int(mask_size[1] * percent)
    #rows = mask_size[1] * percent
    row_distance = int(mask_size[1] / rows)
    #row_distance = mask_size[1] / rows
    distance_crossed = 0
    # cv2.line(mask,(0,distance_crossed),(mask_size[1],distance_crossed),255,1)
    # distance_crossed += 5
    # cv2.line(mask,(0,distance_crossed),(mask_size[1],distance_crossed),255,1)
    for i in range(rows):
        cv2.line(mask, (0, distance_crossed), (mask_size[1], distance_crossed), 1, 1)
        distance_crossed += row_distance
    return mask


def circlePattern(mask_size, radius):
    mask = np.zeros((mask_size[0], mask_size[1]))
    midpoint_x = int((mask_size[0] / 2))
    midpoint_y = int((mask_size[1] / 2))
    cv2.circle(mask, (midpoint_x, midpoint_y), radius, 1, -1)
    return mask


def ellipsePattern(mask_size, major_axis, minor_axis, angle):
    mask = np.zeros((mask_size[0], mask_size[1]))
    midpoint_x = int((mask_size[0] / 2))
    midpoint_y = int((mask_size[1] / 2))
    cv2.ellipse(mask, (midpoint_x, midpoint_y), (major_axis, minor_axis), angle, 0, 360, 1, -1)
    return mask


def bandPattern(mask_size, width, length, angle):
    mask = np.zeros((mask_size[0], mask_size[1]))
    midpoint_x = int((mask_size[0] / 2))
    midpoint_y = int((mask_size[1] / 2))
    startpoint_x = int(midpoint_x - (np.abs(length) / 2))
    startpoint_y = int(midpoint_y + (np.abs(width) / 2))
    endpoint_x = int(midpoint_x + (np.abs(length) / 2))
    endpoint_y = int(midpoint_y - (np.abs(width) / 2))
    cv2.rectangle(mask, (startpoint_x, startpoint_y), (endpoint_x, endpoint_y), 1, -1)
    return mask


def radialPattern(mask_size, ray_count):
    mask = None
    return mask


def spiralPattern(mask_size, sparsity):
    mask = None
    return mask
