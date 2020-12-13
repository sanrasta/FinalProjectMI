import cv2
import numpy as np

import src.project.ImageSynthesisNoise as noise
import src.project.SelectiveImageAcquisition as aqc
import src.project.Utilities as util

cardiac = util.loadImage("images/cardiac.jpg")
util.displayImage(cardiac)

normal_cardiac = util.normalizeImage(cardiac)
print(normal_cardiac)

height, width = cardiac.shape
mask_size = np.array([height, width])
p1mask = aqc.bandPattern(mask_size, 5, 100, 35)
util.displayImage(p1mask)

dft_cardiac = util.getDFT(normal_cardiac)

p1applied = util.applyMask(dft_cardiac, p1mask)

p1image = util.getImage(p1applied)
util.displayImage(p1image)
