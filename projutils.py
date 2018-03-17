"""
Created on Fri Jan  5 20:21:00 2018

@author: Bogdan Kandra

Contains utility functions for the watermarking project
"""
import numpy as np
from matplotlib import pyplot as plt

def set_lsb(number, bit):
    """Sets the Least Significant Bit of a number to the specified bit value.

    Arguments:
        number (int) -- the number to be modified

        bit (int) -- the value to be set as the LSB of number

    Returns:
        int type
    """
    binary = str(bin(number))[2:]
    retString = binary[:-1] + str(bit)
    return int(retString, 2)

def get_lsb(number):
    """Extracts the Least Significant Bit of a number.

    Arguments:
        number (int) -- the number from which the LSB is to be extracted

    Returns:
        int type -- the LSB of the number
    """
    binary = str(bin(number))[2:]
    return int(binary[-1], 2)

def str_to_bin(string):
    """Converts a string of characters into a string
    containing the binary representations of the characters.

    Arguments:
        string (str) -- the string to be converted

    Returns:
        str type
    """
    binaryLetters = list(map(lambda letter: bin(ord(letter))[2:], string))
    return ''.join(map(lambda s: '0' * (8 - len(s)) + s, binaryLetters))

def not_bits(number):
    """Takes a number in string format and performs a not bitwise operation
    on all bits from its binary representation.

    Arguments:
        number (str) -- the number to be modified

    Returns:
        str type
    """
    notBit = lambda c: '1' if c == '0' else '0'
    return ''.join(list(map(notBit, number)))

def fft_plot(image, cmap=None):
    """Takes an image, computes its FFT and displays both using pyplot.

    Arguments:
        image (numPy array) -- the image to be transformed
        
        cmap (str) -- optional, the color map to be used; default value is None

    Returns:
        Nothing
    """
    
    hostFFT = np.fft.fft2(image)    # Take the FFT of the image
    hostShifted = np.fft.fftshift(hostFFT)    # Center it
    # Take the magnitudes and reduce values by logarithming
    magnitudes = np.log(np.abs(hostShifted) + 1)
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(image, cmap)
    plt.subplot(122)
    plt.imshow(magnitudes, cmap)
    
    plt.show()
