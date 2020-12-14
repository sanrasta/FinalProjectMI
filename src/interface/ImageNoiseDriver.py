import cv2
import numpy as np

import src.project.ImageSynthesisNoise as noise
import src.project.SelectiveImageAcquisition as aqc
import src.project.Utilities as util



cardiac = util.loadImage("images/cardiac.jpg")
normal_cardiac = util.normalizeImage(cardiac)
dft_cardiac = util.getDFT(normal_cardiac)
height, width = cardiac.shape
cardiac_size = np.array([height, width])


brain = util.loadImage("images/brain.png")
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
        snr_p6 = util.signalToNoise(brain, p6fimage)
        print(filename, snr_p6)
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



noisy = util.loadMatrix("images/noisyimage.npy")
noisy_2 = noisy.real.astype(np.complex128)
noisy_image = util.getImage(noisy_2)
noisyfi = util.post_process_image(noisy_image)
#util.displayImage(noisyfi)
noisy_2_write = util.writableDFT(noisy_2)
#util.displayImage_plt(noisy_2_write)
height_noisy, width_noisy = noisy_2.shape
noisy_size = np.array([height_noisy, width_noisy])
p8mask1 = noise.butterworthLowpassFilter(noisy_size, 99, 100)
p8mask2 = noise.gaussianLowpassFilter(noisy_size, 50)
p8mask3 = noise.idealLowpassFilter(noisy_size, 99)
p8applied1 = util.applyMask(noisy_2, p8mask1)
p8applied2 = util.applyMask(noisy_2, p8mask2)
p8applied3 = util.applyMask(noisy_2, p8mask3)
p8image1 = util.getImage(p8applied1)
p8image2 = util.getImage(p8applied2)
p8image3 = util.getImage(p8applied3)
p8fimage1 = util.post_process_image(p8image1)
p8fimage2 = util.post_process_image(p8image2)
p8fimage3 = util.post_process_image(p8image3)
util.saveImage("p8_Buttersworth_99_100.jpg", p8fimage1)
util.saveImage("p8_Gaussian_50.jpg", p8fimage1)
util.saveImage("p8_Ideal_99.jpg", p8fimage1)
