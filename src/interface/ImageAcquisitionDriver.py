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
mask_size = np.array([height, width])

p1mask = aqc.bandPattern(mask_size, 5, 100, 35)
util.displayImage(p1mask)

p1applied = util.applyMask(dft_cardiac, p1mask)

p1image = util.getImage(p1applied)
util.displayImage(p1image)



p3mask = aqc.cartesianPattern(mask_size, 50)
util.displayImage(p3mask)

p3applied = util.applyMask(dft_cardiac, p3mask)

p3image = util.getImage(p3applied)
util.displayImage(p3image)



p4mask = aqc.radialPattern(mask_size, 4)
util.displayImage(p4mask)
