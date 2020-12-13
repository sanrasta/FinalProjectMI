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

p1mask = aqc.bandPattern(cardiac_size, 5, 100, 35)
util.displayImage(p1mask)

p1applied = util.applyMask(dft_cardiac, p1mask)

p1image = util.getImage(p1applied)
util.displayImage(p1image)



p3mask = aqc.cartesianPattern(cardiac_size, 50)
util.displayImage(p3mask)

p3applied = util.applyMask(dft_cardiac, p3mask)

p3image = util.getImage(p3applied)
util.displayImage(p3image)



p4mask = aqc.radialPattern(cardiac_size, 8)
util.displayImage(p4mask)

p4applied = util.applyMask(dft_cardiac, p4mask)

p4image = util.getImage(p4applied)
util.displayImage(p4image)
