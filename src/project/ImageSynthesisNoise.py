import numpy as np
import math


def idealLowpassFilter(emptymask, cutoff):
    P = emptymask[0]
    Q = emptymask[1]
    d0 = cutoff

    mask = np.zeros(emptymask) * 255
    for u in range(P):
        for v in range(Q):
            # distance from point(u,v)
            D = (math.sqrt(math.pow(u - P / 2, 2) + math.pow(v - Q / 2, 2)))
            if D <= d0:
                mask[u, v] = 255
            else:
                mask[u, v] = 0
    return mask


def idealHighpassFilter(emptymask, cutoff):
    return 1. - idealLowpassFilter(emptymask, cutoff)


def gaussianLowpassFilter(emptymask, cutoff):
    P = emptymask[0]
    Q = emptymask[1]
    d0 = cutoff
    mask = np.zeros(emptymask) * 255

    for u in range(P):
        for v in range(Q):
            D = (math.sqrt(math.pow(u - P / 2, 2) + math.pow(v - Q / 2, 2)))
            mask[u, v] = math.exp(math.pow(-D, 2) / 2 * (math.pow(d0, 2)))

    return mask


def gaussianHighpassFilter(emptymask, cutoff):
    P = emptymask[0]
    Q = emptymask[1]
    mask = np.zeros(emptymask) * 255
    d0 = cutoff
    for u in range(P):
        for v in range(Q):
            D = (math.sqrt(math.pow(u - P / 2, 2) + math.pow(v - Q / 2, 2)))

            mask[u, v] =  math.exp(math.pow(-D, 2) / 2 * (math.pow(d0, 2)))

    return mask


def butterworthLowpassFilter(emptymask, cutoff, order):
    P = emptymask[0]
    Q = emptymask[1]
    mask = np.zeros(emptymask) * 255
    d0 = cutoff
    n = order
    for u in range(P):
        for v in range(Q):
            D = (math.sqrt(math.pow(u - P / 2, 2) + math.pow(v - Q / 2, 2)))
            # cover divide by 0 scenario
            if D == 0:
                mask[u, v] = 0
            else:
                mask[u, v] = 1 * 255 / 1 + (math.pow((d0 / D), 2 * n))
    return mask


def butterworthHighpassFilter(emptymask, cutoff, order):
    P = emptymask[0]
    Q = emptymask[1]
    mask = np.zeros(emptymask) * 255
    d0 = cutoff
    n = order
    for u in range(P):
        for v in range(Q):
            D = (math.sqrt(math.pow(u - P / 2, 2) + math.pow(v - Q / 2, 2)))
            # cover divide by 0 scenario
            if D == 0:
                mask[u, v] = 0
            else:
                mask[u, v] = 1 * 255 / 1 + (math.pow((D / d0), 2 * n))
    return mask


def ringLowpassFilter(emptymask, cutoff, thickness):
    mask = None
    return mask


def ringHighpassFilter(emptymask, cutoff, thickness):
    mask = None
    return mask
