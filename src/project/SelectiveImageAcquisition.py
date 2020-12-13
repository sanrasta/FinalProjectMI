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
    rows = int(mask_size[0] * (percent/100))
    print(rows)
    #rows = mask_size[1] * percent
    row_distance = int(mask_size[0] / rows)
    print(row_distance)
    #row_distance = mask_size[1] / rows
    distance_crossed = 1
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
    angle = -(90+angle)*np.pi/180
    length = np.abs(length)
    width = np.abs(width)
    midpoint_x = (mask_size[1] / 2)
    midpoint_y = (mask_size[0] / 2)
    pt1_x_float = (midpoint_x - (width / 2))
    pt1_y_float = (midpoint_y + (length / 2))
    pt2_x_float = (midpoint_x - (width / 2))
    pt2_y_float = (midpoint_y - (length / 2))
    pt3_x_float = (midpoint_x + (width / 2))
    pt3_y_float = (midpoint_y - (length / 2))
    pt4_x_float = (midpoint_x + (width / 2))
    pt4_y_float = (midpoint_y + (length / 2))

    # Point 1
    pt1_rotated_x = round(np.cos(angle) * (pt1_x_float - midpoint_x) - np.sin(angle) * (pt1_y_float - midpoint_y) + midpoint_x, 0)
    pt1_rotated_y = round(np.sin(angle) * (pt1_x_float - midpoint_x) + np.cos(angle) * (pt1_y_float - midpoint_y) + midpoint_y, 0)

    # Point 2
    pt2_rotated_x = round(np.cos(angle) * (pt2_x_float - midpoint_x) - np.sin(angle) * (pt2_y_float - midpoint_y) + midpoint_x, 0)
    pt2_rotated_y = round(np.sin(angle) * (pt2_x_float - midpoint_x) + np.cos(angle) * (pt2_y_float - midpoint_y) + midpoint_y, 0)

    # Point 3
    pt3_rotated_x = round(np.cos(angle) * (pt3_x_float - midpoint_x) - np.sin(angle) * (pt3_y_float - midpoint_y) + midpoint_x, 0)
    pt3_rotated_y = round(np.sin(angle) * (pt3_x_float - midpoint_x) + np.cos(angle) * (pt3_y_float - midpoint_y) + midpoint_y, 0)

    # Point 4
    pt4_rotated_x = round(np.cos(angle) * (pt4_x_float - midpoint_x) - np.sin(angle) * (pt4_y_float - midpoint_y) + midpoint_x, 0)
    pt4_rotated_y = round(np.sin(angle) * (pt4_x_float - midpoint_x) + np.cos(angle) * (pt4_y_float - midpoint_y) + midpoint_y, 0)
    #print([pt1_rotated_x, pt1_rotated_y, pt2_rotated_x, pt2_rotated_y, pt3_rotated_x, pt3_rotated_y, pt4_rotated_x,pt4_rotated_y])
    pts = np.array([[pt1_rotated_x, pt1_rotated_y], [pt2_rotated_x, pt2_rotated_y], [pt3_rotated_x, pt3_rotated_y],[pt4_rotated_x, pt4_rotated_y]], dtype=np.int32)
    pts.reshape((-1, 1, 2))
    #print(pts)
    cv2.fillPoly(mask, [pts], 1)

    return mask


def radialPattern(mask_size, ray_count):
    mask = np.zeros((mask_size[0], mask_size[1]))
    midpoint_x = int((mask_size[0] / 2))
    midpoint_y = int((mask_size[1] / 2))
    cv2.line(mask, (0,midpoint_y), (mask_size[0], midpoint_y), 1, 1)

    pt1_x_float = 0
    pt1_y_float = midpoint_y

    pt3_x_float = midpoint_x*2
    pt3_y_float = midpoint_y

    angle = -1*(180/ray_count) * (np.pi / 180)

    iterations = ray_count - 1

    for i in range(1,ray_count):
        # Point 1
        pt1_rotated_x = int(round(np.cos(angle*i) * (pt1_x_float - midpoint_y) - np.sin(angle*i) * (pt1_y_float - midpoint_y) + midpoint_x, 0))
        pt1_rotated_y = int(round(np.sin(angle*i) * (pt1_x_float - midpoint_x) + np.cos(angle*i) * (pt1_y_float - midpoint_y) + midpoint_y, 0))
        print(pt1_rotated_x,pt1_rotated_y)
        # Point 3
        pt3_rotated_x = int(round(np.cos(angle*i) * (pt3_x_float - midpoint_y) - np.sin(angle*i) * (pt3_y_float - midpoint_y) + midpoint_x, 0))
        pt3_rotated_y = int(round(np.sin(angle*i) * (pt3_x_float - midpoint_x) + np.cos(angle*i) * (pt3_y_float - midpoint_y) + midpoint_y, 0))
        print(pt3_rotated_x, pt3_rotated_y)
        cv2.line(mask, (pt1_rotated_x, pt1_rotated_y), (pt3_rotated_x, pt3_rotated_y), 1, 1)

    return mask


def spiralPattern(mask_size, sparsity):
    mask = None
    return mask
