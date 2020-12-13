import cv2
import numpy as np

import src.project.ImageSynthesisNoise as noise
import src.project.SelectiveImageAcquisition as aqc
import src.project.Utilities as util

cardiac = util.loadImage("images/cardiac.jpg")
util.displayImage(cardiac)

normal_cardiac = util.normalizeImage(cardiac)

dft_cardiac = util.getDFT(normal_cardiac)

height, width = cardiac.shape
cardiac_size = np.array([height, width])

brain = util.loadImage("images/brain.png")
util.displayImage(brain)

normal_brain = util.normalizeImage(brain)

dft_brain = util.getDFT(normal_brain)

height, width = brain.shape
brain_size = np.array([height, width])

p6mask = noise.butterworthLowpassFilter(brain_size, 30, 2)
util.displayImage(p6mask)

p6applied = util.applyMask(dft_brain, p6mask)

p6image = util.getImage(p6applied)
util.displayImage(p6image)

p7lmask = noise.gaussianLowpassFilter(brain_size, 100)
util.displayImage(p7lmask)

p7lapplied = util.applyMask(dft_brain, p7lmask)

p7limage = util.getImage(p7lapplied)
util.displayImage(p7limage)

p7hmask = noise.gaussianHighpassFilter(brain_size, 100)
util.displayImage(p7hmask)

p7happlied = util.applyMask(dft_brain, p7hmask)

p7himage = util.getImage(p7happlied)
util.displayImage(p7himage)


