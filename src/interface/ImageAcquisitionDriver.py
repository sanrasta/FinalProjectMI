import cv2
import numpy as np

import src.project.SelectiveImageAcquisition as aqc
import src.project.Utilities as util

cardiac = util.loadImage("images/cardiac.jpg")
util.displayImage(cardiac)
normal_cardiac = util.normalizeImage(cardiac)
dft_cardiac = util.getDFT(normal_cardiac)
#util.saveMatrix("Cardiac_DFT.jpg", dft_cardiac)
height, width = cardiac.shape
cardiac_size = np.array([height, width])


brain = util.loadImage("images/brain.png")
util.displayImage(brain)
normal_brain = util.normalizeImage(brain)
dft_brain = util.getDFT(normal_brain)
#util.saveMatrix("Brain_DFT.jpg", dft_brain)
height, width = brain.shape
brain_size = np.array([height, width])



p1mask = aqc.bandPattern(cardiac_size, 5, 100, 35)
p1fmask = util.post_process_image(p1mask)
util.saveImage("p1fmask.jpg", p1fmask)
p1applied = util.applyMask(dft_cardiac, p1mask)
p1image = util.getImage(p1applied)
p1fimage = util.post_process_image(p1image)
util.saveImage('p1_Masked_Image.jpg', p1fimage)



p2mask1 = aqc.bandPattern(cardiac_size, 15, 128, 10)
p2applied1 = util.applyMask(dft_cardiac, p2mask1)
p2image1 = util.getImage(p2applied1)
p2fimage1 = util.post_process_image(p2image1)
util.saveImage("p2_Masked_Image_1.jpg", p2fimage1)

p2mask2 = aqc.bandPattern(cardiac_size, 10, 171, 135)
p2applied2 = util.applyMask(dft_cardiac, p2mask2)
p2image2 = util.getImage(p2applied2)
p2fimage2 = util.post_process_image(p2image2)
util.saveImage("p2_Masked_Image_2.jpg", p2fimage2)

p2mask3 = aqc.bandPattern(cardiac_size, 30, 100, 270)
p2applied3 = util.applyMask(dft_cardiac, p2mask3)
p2image3 = util.getImage(p2applied3)
p2fimage3 = util.post_process_image(p2image3)
util.saveImage("p2_Masked_Image_3.jpg", p2fimage3)

p2mask4 = aqc.bandPattern(cardiac_size, 50, 200, 300)
p2applied4 = util.applyMask(dft_cardiac, p2mask4)
p2image4 = util.getImage(p2applied4)
p2fimage4 = util.post_process_image(p2image4)
util.saveImage("p2_Masked_Image_4.jpg", p2fimage4)



p3mask = aqc.cartesianPattern(cardiac_size, 50)
p3applied = util.applyMask(dft_cardiac, p3mask)
p3image = util.getImage(p3applied)
p3fimage = util.post_process_image(p3image)
util.saveImage("p3_Masked_Image.jpg", p3fimage)

p3mask10 = aqc.cartesianPattern(cardiac_size, 10)
p3applied10 = util.applyMask(dft_cardiac, p3mask10)
p3image10 = util.getImage(p3applied10)
p3fimage10 = util.post_process_image(p3image10)
util.saveImage("p3_Masked_Image_10%.jpg", p3fimage10)

p3mask20 = aqc.cartesianPattern(cardiac_size, 20)
p3applied20 = util.applyMask(dft_cardiac, p3mask20)
p3image20 = util.getImage(p3applied20)
p3fimage20 = util.post_process_image(p3image20)
util.saveImage("p3_Masked_Image_20%.jpg", p3fimage20)

p3mask33 = aqc.cartesianPattern(cardiac_size, 33)
p3applied33 = util.applyMask(dft_cardiac, p3mask33)
p3image33 = util.getImage(p3applied33)
p3fimage33 = util.post_process_image(p3image33)
util.saveImage("p3_Masked_Image_33%.jpg", p3fimage33)

p3mask75 = aqc.cartesianPattern(cardiac_size, 75)
p3applied75 = util.applyMask(dft_cardiac, p3mask75)
p3image75 = util.getImage(p3applied75)
p3fimage75 = util.post_process_image(p3image75)
util.saveImage("p3_Masked_Image_75%.jpg", p3fimage75)



p4cm8 = aqc.radialPattern(cardiac_size, 8)
p4ca8 = util.applyMask(dft_cardiac, p4cm8)
p4ci8 = util.getImage(p4ca8)
p4fci8 = util.post_process_image(p4ci8)
util.saveImage("p4_Cardiac_Masked_Image_8.jpg", p4fci8)

p4cm6 = aqc.radialPattern(cardiac_size, 6)
p4ca6 = util.applyMask(dft_cardiac, p4cm6)
p4ci6 = util.getImage(p4ca6)
p4fci6 = util.post_process_image(p4ci6)
util.saveImage("p4_Cardiac_Masked_Image_6.jpg", p4fci6)

p4cm9 = aqc.radialPattern(cardiac_size, 9)
p4ca9 = util.applyMask(dft_cardiac, p4cm9)
p4ci9 = util.getImage(p4ca9)
p4fci9 = util.post_process_image(p4ci9)
util.saveImage("p4_Cardiac_Masked_Image_9.jpg", p4fci9)

p4cm20 = aqc.radialPattern(cardiac_size, 20)
p4ca20 = util.applyMask(dft_cardiac, p4cm20)
p4ci20 = util.getImage(p4ca20)
p4fci20 = util.post_process_image(p4ci20)
util.saveImage("p4_Cardiac_Masked_Image_20.jpg", p4fci20)


p4bm8 = aqc.radialPattern(brain_size, 8)
p4ba8 = util.applyMask(dft_brain, p4bm8)
p4bi8 = util.getImage(p4ba8)
p4fbi8 = util.post_process_image(p4bi8)
util.saveImage("p4_Brain_Masked_Image_8.jpg", p4fci8)

p4bm6 = aqc.radialPattern(brain_size, 6)
p4ba6 = util.applyMask(dft_brain, p4bm6)
p4bi6 = util.getImage(p4ba6)
p4fbi6 = util.post_process_image(p4bi6)
util.saveImage("p4_Brain_Masked_Image_6.jpg", p4fbi6)

p4bm9 = aqc.radialPattern(brain_size, 9)
p4ba9 = util.applyMask(dft_brain, p4bm9)
p4bi9 = util.getImage(p4ba9)
p4fbi9 = util.post_process_image(p4bi9)
util.saveImage("p4_Brain_Masked_Image_9.jpg", p4fbi9)

p4bm20 = aqc.radialPattern(brain_size, 20)
p4ba20 = util.applyMask(dft_brain, p4bm20)
p4bi20 = util.getImage(p4ba20)
p4fbi20 = util.post_process_image(p4bi20)
util.saveImage("p4_Brain_Masked_Image_20.jpg", p4fbi20)



p5mask = aqc.radialPattern(brain_size, 20)
p5applied = util.applyMask(dft_brain, p5mask)
p5image = util.getImage(p5applied)
p5fimage = util.post_process_image(p5image)
util.saveImage("p5_Masked_Image.jpg", p5fimage)
