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


cutoff = np.array([5, 20, 45, 60])
order = np.array([1, 2, 3, 4])
for i in cutoff:
    for j in order:
        p6mask = noise.butterworthLowpassFilter(brain_size, i, j)
        p6applied = util.applyMask(dft_brain, p6mask)
        p6image = util.getImage(p6applied)
        p6fimage = util.post_process_image(p6image)
        filename = "p6_Masked_Image_" + str(i) + "_" + str(j) + ".jpg"
        util.saveImage(filename, p6fimage)



glhp = np.array([200, 40, 120, 10])
for k in glhp:
    p7lmask = noise.gaussianLowpassFilter(brain_size, k)
    p7lapplied = util.applyMask(dft_brain, p7lmask)
    p7limage = util.getImage(p7lapplied)
    p7lfimage = util.post_process_image(p7limage)
    filename = "p7_GLP_Masked_Image_" + str(k) + ".jpg"
    util.saveImage(filename, p7lfimage)

    p7hmask = noise.gaussianHighpassFilter(brain_size, k)
    p7happlied = util.applyMask(dft_brain, p7hmask)
    p7himage = util.getImage(p7happlied)
    p7hfimage = util.post_process_image(p7himage)
    filename = "p7_GHP_Masked_Image_" + str(k) + ".jpg"
    util.saveImage(filename, p7hfimage)
