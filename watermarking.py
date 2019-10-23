"""
Created on Sat Dec 30 14:54:23 2017

@author: Bogdan Kandra

Contains functions for embedding a text watermark into a host image
and extracting the watermark from a watermarked image
"""

import numpy as np
import cv2
import pywt
import projutils

def lsb_embed(wmText, hostImage):
    """Embeds a watermark text into a host image, using the LSB technique.\n
    The text length and the text bit representation are embedded
    into the LSB-1 (first least significant bit) of the Red channel of the image.\n
    It assumes that the host image has a width of at least 16 pixels.\n
    It works for watermark texts of maximum 65535 ASCII characters (2 ^ 16 - 1).

    Arguments:
        wmText (str) -- the string to be embedded

        hostImage (NumPy array) -- the image to be watermarked

    Returns:
        NumPy array type -- the watermarked image
    """
    # Compute the watermark text's length and convert it to binary
    wmLength = len(wmText)
    wmLengthBinary = str(bin(wmLength))[2:]
    wmTextBits = wmLength * 8  # Watermark texts' number of bits

    # Extract the red channel and other information from the host image
    #redChannel = hostImage[:,:,2]  # Makes a shallow copy -- a view
    redChannel = cv2.split(hostImage)[2]
    imageH, imageW = hostImage.shape[:2]
    pixelCount = hostImage.size // 3  # Number of pixels
    
    # Check if the wm text's number of bits exceeds that of the host image
    errMessage = """AssertionError: Watermark text length greater than the image
#              number of pixels available. Input watermark text must be shorter
#              or input image size greater"""
    assert wmTextBits <= (3 * pixelCount - 16), errMessage

    # Embed the watermark length into the Red Channel
    # of the first 16 pixels' LSB-1 from the host image
    wmLengthBits = len(wmLengthBinary)
    trailingZeros = 16 - wmLengthBits   # Pad with zero valued LSBs on the left
    i = 0

    for x in range(trailingZeros):
        px = redChannel.item(0, x)       # Get the pixel value
        wmPx = projutils.set_lsb(px, 0)  # Modify LSB of pixel value to 0
        redChannel.itemset((0, x), wmPx) # Update the pixel value

    for x in range(trailingZeros, 16):
        px = redChannel.item(0, x)                       # Get the pixel value
        wmPx = projutils.set_lsb(px, wmLengthBinary[i])  # Modify LSB of pixel value
        redChannel.itemset((0, x), wmPx)                 # Update the pixel value
        i += 1

    # Convert the watermark text to binary
    wmTextBinary = projutils.str_to_bin(wmText)

    # Invert the bits of the watermark text
    wmTextFinal = projutils.not_bits(wmTextBinary)

    # Compute the number of copies of the watermark text to be embedded
    # Treat only the case where wmTextBits <= pixelCount - 16
    # <TODO> The cases when wmTextBits > I - 16 and wmTextBits > 2 * I - 16
    i = 0

    # Embed the watermark text into the Red Channel
    # of the pixels' LSB-1 from the host image
    for x in range(imageH):
        for y in range(imageW):
            if x == 0 and y <= 15:
                # the first 16 pixels have been embedded with the watermark length
                continue
            else:
                px = redChannel.item(x, y)
                wmPx = projutils.set_lsb(px, wmTextFinal[i])
                redChannel.itemset((x, y), wmPx)
                i += 1
                if i >= wmTextBits:
                    i = i % wmTextBits

    # Build the watermarked image
    wmImage = cv2.merge([hostImage[:,:,0], hostImage[:,:,1], redChannel])

    return wmImage

def lsb_detect(wmImage):
    """Detects the text embedded into the LSB-1 (first least significant bit)
    of the Red Channel pixels of the input image.\n
    The first 16 Red Channel values contain the watermark text length,
    followed by the watermark text bit representation.\n
    It assumes that the host image has a width of at least 16 pixels.\n
    It works for watermark texts of maximum 65535 ASCII characters (2 ^ 16 - 1).

    Arguments:
        wmImage (NumPy array) -- the image to be processed for detection

    Returns:
        str type -- the embedded text
    """
    redChannel = cv2.split(wmImage)[2]
    imageH, imageW = wmImage.shape[:2]

    # Take the first 16 Red Channel pixel LSBs
    wmLengthBinary = ''

    for x in range(16):
        px = redChannel.item(0, x)
        pxLSB = projutils.get_lsb(px)
        wmLengthBinary += str(pxLSB)

    # Take the next pixels as to recover the embedded text
    wmTextLength = int(wmLengthBinary, 2)
    wmTextRecovered = ''
    letterBinary = ''
    bitsRead = 0

    for x in range(imageH):
        for y in range(imageW):
            if x == 0 and y <= 15:
                # the first 16 pixels have been embedded with the watermark length
                continue
            elif len(wmTextRecovered) == wmTextLength:
                x = imageH - 1
                y = imageW - 1
                continue
            else:
                if bitsRead == 8:  # We have read a letter, prepare the next one
                    letterInversed = projutils.not_bits(letterBinary)
                    wmLetterInt = int(letterInversed, 2)
                    wmTextRecovered += chr(wmLetterInt)
                    letterBinary = ''
                    bitsRead = 0
                px = redChannel.item(x, y)
                pxLSB = projutils.get_lsb(px)
                letterBinary += str(pxLSB)
                bitsRead += 1

    return wmTextRecovered

def fft_embed(wmImage, hostImage, alpha):
    """Embeds a watermark image into a host image, using the Fast Fourier 
    Transform.\n
    
    Arguments:
        wmImage (NumPy array) -- the image to be embedded

        hostImage (NumPy array) -- the image to be watermarked
        
        alpha (float) -- the embedding strength factor

    Returns:
        NumPy array type -- the watermarked image, in float64 format
    """
    # Take the dimensions of the host and watermark images
    wmH, wmW = wmImage.shape[:2]
    hostH, hostW = hostImage.shape[:2]

    # Resize the watermark image so that it is the same size as the host image
    if wmH > hostH or wmW > hostW:
        # Scale down the watermark image
        wmImage = cv2.resize(wmImage, (hostW, hostH), interpolation = cv2.INTER_AREA)
    elif wmH < hostH or wmW < hostW:
        # Scale up the watermark image
        wmImage = cv2.resize(wmImage, (hostW, hostH), interpolation = cv2.INTER_LINEAR)

    # Take the new dimensions of the watermark image
    wmH, wmW = wmImage.shape[:2]

    # Take the FFT of the host image
    hostFFT = np.fft.fft2(hostImage)
    hostShifted = np.fft.fftshift(hostFFT)     # Center it
    
    # Take the FFT of the watermark image
    wmFFT = np.fft.fft2(wmImage)
    wmShifted = np.fft.fftshift(wmFFT)
    
    # Generate result complex matrix
    resultShifted = np.zeros(wmFFT.shape, dtype='complex128')

    # Watermark the host image using the frequency domain
    resultShifted = hostShifted + alpha * wmShifted
    # Take its Inverse FFT and convert the resulting values into floats
    resultFFT = np.fft.ifftshift(resultShifted)
    resultImage = np.fft.ifft2(resultFFT)
    resultImage = np.real(resultImage)

    return resultImage

def fft_detect(resultImage, hostImage, alpha):
    """Starting from a watermarked image, detects the watermark image embedded 
    into the host image by using the Fast Fourier Transform.\n
    
    Arguments:
        resultImage (NumPy array) -- the watermarked image

        hostImage (NumPy array) -- the original image
        
        alpha (float) -- the embedding strength factor

    Returns:
        NumPy array type -- the watermark image
    """
    # Take the FFT of the watermarked image
    resultFFT = np.fft.fft2(resultImage)
    resultShifted = np.fft.fftshift(resultFFT)    # Center it

    # Take the FFT of the host image
    hostFFT = np.fft.fft2(hostImage)
    hostShifted = np.fft.fftshift(hostFFT)    # Center it
    
    # Generate watermark complex matrix
    wmShifted = np.zeros(hostFFT.shape, dtype='complex128')
    
    # Compute the watermark image using the frequency domain
    wmShifted = (resultShifted - hostShifted) / alpha
    # Take its Inverse FFT and convert the resulting values into floats
    wmFFT = np.fft.ifftshift(wmShifted)
    wmImage = np.fft.ifft2(wmFFT)
    wmImage = np.rint(np.real(wmImage))
    wmImage = np.uint8(wmImage)
    
    return wmImage

def dwt_embed(wmImage, hostImage, alpha, beta):
    """Embeds a watermark image into a host image, using the First Level 
    Discrete Wavelet Transform and Alpha Blending.\n
    The formula used for the alpha blending is:
        resultLL = alpha * hostLL + beta * watermarkLL

    Arguments:
        wmImage (NumPy array) -- the image to be embedded

        hostImage (NumPy array) -- the image to be watermarked
        
        alpha (float) -- the first embedding strength factor
        
        beta (float) -- the second embedding strength factor

    Returns:
        NumPy array type -- the watermarked image, in float64 format
    """
    # Take the dimensions of the host and watermark images
    wmHeight, wmWidth = wmImage.shape[:2]
    hostHeight, hostWidth = hostImage.shape[:2]
    
    # Resize the watermark image so that it is the same size as the host image
    if wmHeight > hostHeight or wmWidth > hostWidth:
        # Scale down the watermark image
        wmImage = cv2.resize(wmImage, (hostWidth, hostHeight), interpolation = cv2.INTER_AREA)
    elif wmHeight < hostHeight or wmWidth < hostWidth:
        # Scale up the watermark image
        wmImage = cv2.resize(wmImage, (hostWidth, hostHeight), interpolation = cv2.INTER_LINEAR)
    
    # Take the new dimensions of the watermark image
    wmHeight, wmWidth = wmImage.shape[:2]
    
    # Split both images into channels
    hostB, hostG, hostR = cv2.split(hostImage)
    wmB, wmG, wmR = cv2.split(wmImage)
    
    # Compute the first level bidimensional DWT for each channel of both images
    # (LL, (HL, LH, HH))
    cAhostB, (cHhostB, cVhostB, cDhostB) = pywt.dwt2(hostB, 'db2')
    cAhostG, (cHhostG, cVhostG, cDhostG) = pywt.dwt2(hostG, 'db2')
    cAhostR, (cHhostR, cVhostR, cDhostR) = pywt.dwt2(hostR, 'db2')

    cAhostHeight, cAhostWidth = cAhostB.shape

    cAwmB, (cHwmB, cVwmB, cDwmB) = pywt.dwt2(wmB, 'db2')
    cAwmG, (cHwmG, cVwmG, cDwmG) = pywt.dwt2(wmG, 'db2')
    cAwmR, (cHwmR, cVwmR, cDwmR) = pywt.dwt2(wmR, 'db2')

    cAwmHeight, cAwmWidth = cAwmB.shape
    
    # Generate image matrix for containing all four host coefficients images
    coeffsHost = np.zeros((cAhostHeight * 2, cAhostWidth * 2, 3), dtype = 'float64')

    # Merge channels for each of A, H, V and D and build the host coefficients image
    cAhost = cv2.merge([cAhostB, cAhostG, cAhostR])
    coeffsHost[0:cAhostHeight, 0:cAhostWidth] = cAhost
    cHhost = cv2.merge([cHhostB, cHhostG, cHhostR])
    coeffsHost[0:cAhostHeight, cAhostWidth:cAhostWidth * 2] = cHhost
    cVhost = cv2.merge([cVhostB, cVhostG, cVhostR])
    coeffsHost[cAhostHeight:cAhostHeight * 2, 0:cAhostWidth] = cVhost
    cDhost = cv2.merge([cDhostB, cDhostG, cDhostR])
    coeffsHost[cAhostHeight:cAhostHeight * 2, cAhostWidth:cAhostWidth * 2] = cDhost
    
    # Display the host coefficients image
    temp = np.uint8(np.rint(coeffsHost))
    cv2.imshow('Host DWT', temp)
    
    # Generate image matrix for containing all four watermark coefficients images
    coeffsWm = np.zeros((cAwmHeight * 2, cAwmWidth * 2, 3), dtype = 'float64')

    # Merge channels for each of A, H, V and D and build the wm coefficients image
    cAwm = cv2.merge([cAwmB, cAwmG, cAwmR])
    coeffsWm[0:cAwmHeight, 0:cAwmWidth] = cAwm
    cHwm = cv2.merge([cHwmB, cHwmG, cHwmR])
    coeffsWm[0:cAwmHeight, cAwmWidth:cAwmWidth * 2] = cHwm
    cVwm = cv2.merge([cVwmB, cVwmG, cVwmR])
    coeffsWm[cAwmHeight:cAwmHeight * 2, 0:cAwmWidth] = cVwm
    cDwm = cv2.merge([cDwmB, cDwmG, cDwmR])
    coeffsWm[cAwmHeight:cAwmHeight * 2, cAwmWidth:cAwmWidth * 2] = cDwm

    # Display the watermark coefficients image
    temp = np.uint8(np.rint(coeffsWm))
    cv2.imshow('Watermark DWT', temp)

    # Apply the Alpha Blending Technique
    # wmImageLL = alpha * hostLL + beta * wmLL
    cAresult = alpha * cAhost + beta * cAwm

    cAresultB, cAresultG, cAresultR = cv2.split(cAresult)

    # Compute the channels of the watermarked image by applying the inverse DWT
    resultB = pywt.idwt2((cAresultB, (cHhostB, cVhostB, cDhostB)), 'db2')
    resultG = pywt.idwt2((cAresultG, (cHhostG, cVhostG, cDhostG)), 'db2')
    resultR = pywt.idwt2((cAresultR, (cHhostR, cVhostR, cDhostR)), 'db2')

    # Merge the channels and obtain the final watermarked image
    resultImage = cv2.merge([resultB, resultG, resultR])

    return resultImage

def dwt_detect(resultImage, hostImage, alpha, beta):
    """Starting from a watermarked image, detects the watermark image embedded 
    into the host image by using the Discrete Wavelet Transform and Alpha 
    Blending, with embedding factors alpha and beta.\n
    
    Arguments:
        resultImage (NumPy array) -- the watermarked image

        hostImage (NumPy array) -- the original image
        
        alpha (float) -- the first embedding strength factor
        
        beta (float) -- the second embedding strength factor

    Returns:
        NumPy array type -- the watermark image
    """
#    resultImage = np.uint8(np.rint(resultImage))
    resultHeight, resultWidth = resultImage.shape[:2]

#    # Remove the last extra row of the result image generated by the inverse DWT
#    resultImage = resultImage[0:resultHeight, 0:resultWidth]

    # Split both images into channels
    hostB, hostG, hostR = cv2.split(hostImage)
    resultB, resultG, resultR = cv2.split(resultImage)

    # Compute the first level bidimensional DWT for each channel of both images
    # (LL, (HL, LH, HH))
    cAhostB, (cHhostB, cVhostB, cDhostB) = pywt.dwt2(hostB, 'db2')
    cAhostG, (cHhostG, cVhostG, cDhostG) = pywt.dwt2(hostG, 'db2')
    cAhostR, (cHhostR, cVhostR, cDhostR) = pywt.dwt2(hostR, 'db2')
    cAhost = cv2.merge([cAhostB, cAhostG, cAhostR])

    cAhostHeight, cAhostWidth = cAhostB.shape

    cAresultB, (cHresultB, cVresultB, cDresultB) = pywt.dwt2(resultB, 'db2')
    cAresultG, (cHresultG, cVresultG, cDresultG) = pywt.dwt2(resultG, 'db2')
    cAresultR, (cHresultR, cVresultR, cDresultR) = pywt.dwt2(resultR, 'db2')
    cAresult = cv2.merge([cAresultB, cAresultG, cAresultR])

    cAwmHeight, cAwmWidth = cAresultB.shape

    # Apply the Alpha Blending Technique
    # wmLL = (wmImageLL - alpha * hostLL) / beta
    cAwm = (cAresult - (alpha * cAhost)) / beta

    cAwmB, cAwmG, cAwmR = cv2.split(cAwm)

    # Compute the channels of the watermark image by applying the inverse DWT
    wmB = pywt.idwt2((cAwmB, (cHhostB, cVhostB, cDhostB)), 'db2')
    wmG = pywt.idwt2((cAwmG, (cHhostG, cVhostG, cDhostG)), 'db2')
    wmR = pywt.idwt2((cAwmR, (cHhostR, cVhostR, cDhostR)), 'db2')
    #wmB = pywt.idwt2((cAwmB, (cHresultB, cVresultB, cDresultB)), 'db2')
    #wmG = pywt.idwt2((cAwmG, (cHresultG, cVresultG, cDresultG)), 'db2')
    #wmR = pywt.idwt2((cAwmR, (cHresultR, cVresultR, cDresultR)), 'db2')

    # Merge the channels and obtain the final recovered image
    wmImage = cv2.merge([wmB, wmG, wmR])
    return wmImage
