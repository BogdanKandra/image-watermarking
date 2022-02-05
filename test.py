import cv2
import numpy as np
import tifffile as tiff

hostImagePath = 'images/Trump.jpg'
hostImage = cv2.imread(hostImagePath)

hostFFT = np.fft.fft2(hostImage)
hostShifted = np.fft.fftshift(hostFFT)     # Center it

resultFFT = np.fft.ifftshift(hostShifted)
resultImage = np.fft.ifft2(resultFFT)

resultImages = np.real(resultImage)

tiff.imsave('new.tif', resultImages)
a = tiff.imread('new.tif')

#cv2.imshow('sss', resultImage)

cv2.waitKey()
cv2.destroyAllWindows()