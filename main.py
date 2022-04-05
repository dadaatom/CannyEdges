import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, utils
import math

def orientation(xImage, yImage):
    newImage = np.zeros(shape=(xImage.shape[0],xImage.shape[1],3), dtype=np.uint8)
    for x in range(xImage.shape[0]):
        for y in range(xImage.shape[1]):
            if len(xImage.shape) == 3:
                for k in range(3):
                    if xImage[x, y, 0] > 0:
                        angle = math.atan(yImage[x, y, k] / xImage[x, y, k])
                        newImage[x,y,k] = truncate(int(255*(angle+math.pi/2) / math.pi))
            elif xImage[x, y] > 0:
                angle = math.atan(int(yImage[x, y]) / int(xImage[x, y]))
                newImage[x,y] = truncate(int(255*(angle+math.pi/2) / math.pi))
    return newImage

def magnitude(xImage, yImage):
    newImage = np.zeros(shape=xImage.shape, dtype=np.uint8)
    max = 0;
    for x in range(newImage.shape[0]):
        for y in range(newImage.shape[1]):
            if len(newImage.shape) == 3:
                for i in range(newImage.shape[2]):
                    newImage[x, y] = truncate(math.sqrt(int(xImage[x, y, i]) * int(xImage[x, y, i]) + int(yImage[x, y, i]) * int(yImage[x, y, i])))
            else:
                newImage[x, y] = truncate(math.sqrt(int(xImage[x, y])*int(xImage[x, y]) + int(yImage[x, y])*int(yImage[x, y])))

    return newImage

def createGaussian(size=5, sigma = 1):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def gaussianDifference(image, size, A, B):
    gaussianA = applyFilter(image, normaliseFilter(createGaussian(size, A)))
    gaussianB = applyFilter(image, normaliseFilter(createGaussian(size, B)))

    newImage = np.zeros(image.shape, dtype=np.uint8)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if len(image.shape) == 3:
                for i in range(image.shape[2]):
                    newImage[x,y,i] = truncate(abs(int(gaussianA[x,y,i])-int(gaussianB[x,y,i])))
            else:
                newImage[x, y] = truncate(abs(int(gaussianA[x, y]) - int(gaussianB[x, y])))
    return newImage

def applyFilter(image, filterMat):
    xlen = image.shape[0]
    ylen = image.shape[1]

    xStart = int(filterMat.shape[0] / 2)
    xEnd = int(filterMat.shape[0] / 2 + .5)
    yStart = int(filterMat.shape[1] / 2)
    yEnd = int(filterMat.shape[1] / 2 + .5)

    newImage = np.zeros(image.shape, dtype=np.uint8)

    for i in range(xStart, xlen-xEnd):
        for j in range(yStart, ylen-yEnd):
            if len(image.shape) == 3: #checks if its RGB image or grayscale.
                for k in range(image.shape[2]):
                    newImage[i, j, k] = truncate(np.sum(image[i - xStart: i + xEnd, j - yStart: j + yEnd, k] * filterMat))
            else:
                newImage[i, j] = truncate(np.sum(image[i - xStart: i + xEnd, j - yStart: j + yEnd] * filterMat))
    return newImage

def grayscale(image):
    newImage = np.zeros(shape=image.shape)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            avg = np.sum(image[x,y,:])/3
            newImage[x,y,:] = avg
    return newImage

def normaliseFilter(filter):
    sum = np.sum(filter)
    if sum == 0:
        return filter
    return ((1/sum)*filter)

def truncate(val):
    if val < 0:
        return 0
    elif val > 255:
        return 255
    return val


def display(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)


# ===================== Assignment Part Executions ===================== #

def partTwo(image): #Applies guassian filter to an image and computes difference of gaussian filtered images.


def partThree(image): #Gets magnitude and orientation of an image from scratch. Compares with the open cv canny function.



# ===================== IMAGES ===================== #

ladybugImage = "Images/ladybug.png"
duckImage = "Images/duck.jpg"
santoriniImage = "Images/santorini.jpg"


# ===================== Filters ===================== #

sharpenFilter = np.array([[0,0,0],[0,2,0],[0,0,0]], dtype=np.float64)


# ===================== Execution ===================== #
if __name__ == '__main__':
    imagePath = duckImage

    image = cv2.imread(imagePath)
    image.astype('uint8')
    
    grayscaleImage = cv2.imread(imagePath,0)
    grayscaleImage.astype('uint8')

    display('original', image)


    print('Computing Gaussian Blur...')
    display('gaussian', applyFilter(image, normaliseFilter(createGaussian(10, 10))))
    display('gaussian difference', gaussianDifference(image, 5, 5, .5))


    print('Computing Canny Lines...')
    xSobelFilter = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]], dtype=np.float64)
    ySobelFilter = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]], dtype=np.float64)

    xSobelImage = applyFilter(grayscaleImage, xSobelFilter)
    ySobelImage = applyFilter(grayscaleImage, ySobelFilter)

    display('x sobel', xSobelImage)
    display('y sobel', ySobelImage)

    display('orientation', orientation(xSobelImage, ySobelImage))
    display('magnitude', magnitude(xSobelImage, ySobelImage))
    display('canny', cv2.Canny(grayscaleImage, 100, 200))

    print('Done!')

    cv2.waitKey(0)
