import numpy as np
import math
import cv2


def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def idealLowpassFilter(emptymask, cutoff):
    rows, cols = emptymask
    d0 = cutoff
    center = (rows / 2, cols / 2)
    mask = np.zeros(emptymask) * 255
    for u in range(rows):
        for v in range(cols):
            # distance center to point(u,v)
            D = distance((u, v), center)
            if D <= d0:
                mask[u, v] = 255
            else:
                mask[u, v] = 0
    return mask


def idealHighpassFilter(emptymask, cutoff):
    return 1 - idealLowpassFilter(emptymask, cutoff)


def gaussianLowpassFilter(emptymask, cutoff):
    mask = np.zeros(emptymask) * 255
    rows, cols = emptymask
    center = (rows / 2, cols / 2)
    for u in range(cols):
        for v in range(rows):
            mask[u, v] = math.exp(((-distance((u, v), center) ** 2) / (2 * (cutoff ** 2))))
    return mask


def gaussianHighpassFilter(emptymask, cutoff):
    mask = np.zeros(emptymask) * 255
    rows, cols = emptymask
    center = (rows / 2, cols / 2)
    for u in range(cols):
        for v in range(rows):
            mask[u, v] = 1 - math.exp(((-distance((u, v), center) ** 2) / (2 * (cutoff ** 2))))
    return mask


def butterworthLowpassFilter(emptymask, cutoff, order):
    mask = np.zeros(emptymask[:2])
    rows, cols = emptymask[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            mask[y, x] = 1 / (1 + (distance((y, x), center) / cutoff) ** (2 * order))
    return mask


def butterworthHighpassFilter(emptymask, cutoff, order):
    mask = np.zeros(emptymask[:2])
    rows, cols = emptymask[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            mask[y, x] = 1 * 255 - 1 / (1 + (distance((y, x), center) / cutoff) ** (2 * order))
    return mask

def ringLowpassFilter(emptymask, cutoff, thickness):
    rows, cols = emptymask
    xrow, ycol = rows / 2, cols / 2
    ring_mask = np.zeros(emptymask)

    for u in range(ring_mask.shape[0]):
        for v in range(ring_mask.shape[1]):
            pt = np.sqrt((np.square(u - xrow)) + np.square(v - ycol))
            if (pt <= cutoff + thickness) and (pt > (cutoff - thickness)):
                ring_mask[u][v] = 255
            else:
                ring_mask[u][v] = 0
    mask = ring_mask
    return mask


def ringHighpassFilter(emptymask, cutoff, thickness):
    rows, cols = emptymask
    xrow, ycol = rows / 2, cols / 2
    ring_mask = np.zeros(emptymask)

    for u in range(ring_mask.shape[0]):
        for v in range(ring_mask.shape[1]):
            pt = np.sqrt((np.square(u - xrow)) + np.square(v - ycol))
            if (pt > cutoff + thickness) and (pt <= (cutoff - thickness)):
                ring_mask[u][v] = 0
            else:
                ring_mask[u][v] = 255
    mask = ring_mask
    return mask

