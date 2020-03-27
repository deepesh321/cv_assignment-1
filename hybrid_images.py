import numpy
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import misc
from scipy import ndimage
import math


def makeGaussianFilter(numRows, numCols, sigma, highPass=True):
    # Constructing a Gaussian Filter
    centerI = int(numRows/2) + 1 if numRows % 2 == 1 else int(numRows/2)
    centerJ = int(numCols/2) + 1 if numCols % 2 == 1 else int(numCols/2)

    def gaussian(i, j):
        coefficient = math.exp(-1.0 * ((i - centerI) **
                                       2 + (j - centerJ)**2) / (2 * sigma**2))
        return 1 - coefficient if highPass else coefficient
    # sample values from a spherical gaussian function from the center of the image
    return numpy.array([[gaussian(i, j) for j in range(numCols)] for i in range(numRows)])


def filterDFT(imageMatrix, filterMatrix):
    # Changing the image from spacial domain to frequency domain
    shiftedDFT = fftshift(fft2(imageMatrix))
# Multiplying in frequency domain (which is same as convolution in spacial domain)
    filteredDFT = shiftedDFT * filterMatrix
# Changing the image from frequency domain to spacial domain
    return ifft2(ifftshift(filteredDFT))


def lowPass(imageMatrix, sigma):
    # shape of the image is stored in 'n' and 'm'
    n, m = imageMatrix.shape
    return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=False))


def highPass(imageMatrix, sigma):
    # shape of the image is stored in 'n' and 'm'
    n, m = imageMatrix.shape
    return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=True))


def hybridImage(highFreqImg, lowFreqImg, alpha, beta):
    # Calling the high pass filter with the first image argument with its corresponding threshold value alpha
    highPassed = highPass(highFreqImg, alpha)
    # Calling the low pass filter with the second image argument with its corresponding threshold value beta
    lowPassed = lowPass(lowFreqImg, beta)

    # Returning the sum of the resultants of the images which are passed through the high pass and low pass filters respectively
    return highPassed + lowPassed


# Reading the images using ndimage imported from scipy
x = ndimage.imread("/home/deepesh/Downloads/CV/CV/fish.bmp",
                   flatten=True)  # Image 1
y = ndimage.imread(
    "/home/deepesh/Downloads/CV/CV/submarine.bmp", flatten=True)  # Image 2
alpha = 3  # Can be changed by the user
beta = 7  # Can be changed by the user

# Calling the function Hybrid with the two images read as arguments and passing alpha and beta which are the thresholds for high pass and low pass filter respectively.
hybrid = hybridImage(x, y, alpha, beta)  # Alpha and Beta
# Using misc to save the resultant image from the hybrid function
misc.imsave("fish-submarine.png", numpy.real(hybrid))
