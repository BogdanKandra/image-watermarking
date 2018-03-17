"""
Created on Fri Jan  5 23:35:11 2018

@author: Bogdan Kandra

Watermarking Project Driver Program
"""

import cv2
import watermarking as wm
import projutils as pj
import numpy as np

def lsb_test():
    """Runs a test for the Least Significant Bit watermarking technique.

    Arguments:
        None

    Returns:
        Nothing
    """
    # Take the watermark text
    wmText = 'Criptografie'
    # Take the host image
    imagePath = 'images/lena.tiff'
    hostImage = cv2.imread(imagePath)
    # Compute the host image's size
    imageH, imageW = hostImage.shape[:2]
    
    # Call the embedding procedure
    wmImage = wm.lsb_embed(wmText, hostImage)
    
    # Compute the difference between the watermarked and the unwatermarked images
    diff = cv2.absdiff(hostImage, wmImage)
    for px in range(imageH):
        for py in range(imageW):
            if diff.item(px, py, 2) > 0:   # if there is a difference, augment it
                diff.itemset((px, py, 2), 150)
    
    cv2.imshow('Diferenta', diff)
    cv2.imshow('Initiala', hostImage)
    cv2.imshow('Watermarked', wmImage)
    
#    # Save the watermarked image
#    saved = cv2.imwrite('images/outputs/resultLSB.png', wmImage)
#    if saved == False:
#        print("Eroare la scrierea imaginii watermarked!")
#    
#    # Save the difference image
#    saved = cv2.imwrite('images/outputs/diffLSB.png', diff)
#    if saved == False:
#        print("Eroare la scrierea imaginii diferenta!")
    
    recoveredText = wm.lsb_detect(wmImage)
    
    print('The text embedded into the image is: ' + recoveredText)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def visible_image_test():
    """Runs a test for the visible watermarking technique over an image.

    Arguments:
        None

    Returns:
        Nothing
    """
    wmImagePath = 'images/fmicrypto.png'
    hostImagePath = 'images/lena.tiff'
    
    # Set the flag in order not to lose the Alpha Channel
    wmImage = cv2.imread(wmImagePath, cv2.IMREAD_UNCHANGED)
    
    # Read the host image
    hostImage = cv2.imread(hostImagePath)

    # Call the watermarking function
    output = wm.visible_watermark2(wmImage, hostImage)

    cv2.imshow('Initiala', hostImage)
    cv2.imshow('Watermarked', output)
    
#    # Save the watermarked image
#    saved = cv2.imwrite('images/outputs/resultVisible.png',output)
#    if saved == False:
#        print("Eroare la scrierea imaginii watermarked!")
    
    cv2.waitKey()
    cv2.destroyAllWindows()

def visible_video_test():
    """ Runs a test for the visible watermarking technique over a video.
    
    Arguments:
        None
    
    Returns:
        Nothing
    """
    wmImagePath = 'images/fmicrypto.png'
    videoPath = 'videos/big_buck_bunny.mp4'

    # Set the flag in order not to lose the Alpha Channel
    wmImage = cv2.imread(wmImagePath, cv2.IMREAD_UNCHANGED)

    # Read the video
    stream = cv2.VideoCapture(videoPath)

    success = True
    # Define the VideoWriter object and parameterize it
    # MJPG codec and .avi extension needed for color video
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    result = cv2.VideoWriter('videos/resultVideo.avi', fourcc, 25, (1280, 720))    
    
    while stream.isOpened():
        success, frame = stream.read()  # Read a frame
        if success == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert it to RGB
        # Why is the color of the wm alternating between blue and red?
        #wmImage = cv2.cvtColor(wmImage, cv2.COLOR_RGBA2BGRA)
        frame = wm.visible_watermark1(wmImage, frame)   # Watermark it
        result.write(frame)   # Write it into the result video
    
    stream.release()
    result.release()
    cv2.destroyAllWindows()

def fft_test():
    """ Runs a test for the frequency domain watermarking technique using the 
    Fast Fourier Transform.
    
    Arguments:
        None
    
    Returns:
        Nothing
    """
    wmImagePath = 'images/obama2.jpg'
    hostImagePath = 'images/Trump.jpg'
    alphaFactor = 0.01

    # Read the watermark and the host images
    wmImage = cv2.imread(wmImagePath)
    hostImage = cv2.imread(hostImagePath)
    hostImageGray = cv2.imread(hostImagePath, 0)
    
    # Call the FFT watermarking function
    resultImage = wm.fft_embed(wmImage, hostImage, alphaFactor)

    # Display the host image and the watermarked image
    cv2.imshow('Before', hostImage)
    # Convert the watermarked image from float64 to uint8, for displaying it
    temp = np.uint8(np.rint(resultImage))
    cv2.imshow('After', temp)

    # Compute and display the difference between the images
    diff = cv2.absdiff(temp, hostImage)
    for px in range(temp.shape[0]):
        for py in range(temp.shape[1]):
            # If there are differences, augment them
            if diff.item(px, py, 0) > 0:
                diff.itemset((px, py, 0), diff.item(px, py, 0) + 150)
            if diff.item(px, py, 1) > 0:
                diff.itemset((px, py, 1), diff.item(px, py, 1) + 150)
            if diff.item(px, py, 2) > 0:
                diff.itemset((px, py, 2), diff.item(px, py, 2) + 150)
    cv2.imshow('Difference', diff)

    # Detect the watermark image
    detectedImage = wm.fft_detect(resultImage, hostImage, alphaFactor)

    cv2.imshow('Detected', detectedImage)

    cv2.waitKey()
    cv2.destroyAllWindows()

#    # Save the watermarked image
#    saved = cv2.imwrite('images/outputs/resultFFT.png', resultImage)
#    if saved == False:
#        print("Eroare la scrierea imaginii watermarked!")
#
#    # Save the differences image
#    saved = cv2.imwrite('images/outputs/diffFFT.png', diff)
#    if saved == False:
#        print("Eroare la scrierea imaginii diferenta!")

    # Convert host image to RGB so that it displays properly in pyplot
    hostImage = cv2.cvtColor(hostImage, cv2.COLOR_BGR2RGB)

    # Display the color host image and its FFT
    pj.fft_plot(hostImage)

    # Display the grayscale host image and its FFT
    pj.fft_plot(hostImageGray, 'gray')

def dwt_test():
    """ Runs a test for watermarking using the First Level Discrete Wavelet 
    Transform and Alpha Blending techniques.

    Arguments:
        None

    Returns:
        Nothing
    """
    wmImagePath = 'images/obama.jpg'
    hostImagePath = 'images/lena.tiff'
    alphaFactor = 0.99
    betaFactor = 0.009

    # Read the watermark and the host images
    wmImage = cv2.imread(wmImagePath)
    hostImage = cv2.imread(hostImagePath)

    resultImage = wm.dwt_embed(wmImage, hostImage, alphaFactor, betaFactor)
    resultHeight, resultWidth = resultImage.shape[:2]

    # Display the host image, watermark and the watermarked image
    cv2.imshow('Host Image', hostImage)
    cv2.imshow('Watermark', wmImage)
    # Convert the watermarked image from float64 to uint8, for displaying it
    temp = np.uint8(np.rint(resultImage))
    cv2.imshow('Watermarked Image', temp)

    # Compute and display the difference between the images
    diff = cv2.absdiff(temp, hostImage)
    for px in range(temp.shape[0]):
        for py in range(temp.shape[1]):
            # If there are differences, augment them
            if diff.item(px, py, 0) > 0:
                diff.itemset((px, py, 0), diff.item(px, py, 0) + 150)
            if diff.item(px, py, 1) > 0:
                diff.itemset((px, py, 1), diff.item(px, py, 1) + 150)
            if diff.item(px, py, 2) > 0:
                diff.itemset((px, py, 2), diff.item(px, py, 2) + 150)
    cv2.imshow('Difference', diff)

    recoveredImage = wm.dwt_detect(temp, hostImage, alphaFactor, betaFactor)
    temp2 = np.uint8(np.rint(recoveredImage))
    cv2.imshow('Recovered Image', temp2)

    cv2.waitKey()
    cv2.destroyAllWindows()

#lsb_test()
#visible_image_test()
#visible_video_test()
#fft_test()
dwt_test()