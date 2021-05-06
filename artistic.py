# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:19:33 2019

@author: mhamz
"""

import numpy as np
import math
from PIL import Image
import cv2 
from skimage import data
import matplotlib.pyplot as plt
hiddenimports=['pywt._extensions._cwt']

"Image Imports"

monkey = cv2.imread("lemonke.png")
candy = cv2.imread("candy.jpg")

        
"CONVOLUTION CONVERTER"

def converter(imgin):
    
    imgin2 = imgin.copy()
 
    for x in range(imgin.shape[0]):
        for j in range(imgin.shape[1]):
            imgin2[x][j] = imgin[imgin.shape[0]-x-1][imgin.shape[1]-j-1]
    print(imgin2)
    return imgin2

"FILTER GENERATOR"

def gfiltergen(r):
    
    sigma = r
    fsize = 2 * int(4 * sigma + 0.5) + 1
    gfilter = np.zeros(shape = (fsize, fsize))
    m = fsize//2
    n = fsize//2
        
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gfilter[x+m, y+n] = (1/x1)*x2
    
    return gfilter

"ACTUAL CONVOLUTION"

def actconv(imgin, ker):
    
    ker = converter(ker)
    height = imgin.shape[0]
    width = imgin.shape[1]
    kheight = ker.shape[0]
    kwidth = ker.shape[1]
    hf = kheight//2
   
    wf = kwidth//2
   
    
    imgconv = np.zeros(imgin.shape)
    for x in range(hf, height-hf):
        for j in range(wf, width-wf):
            
            total = 0
            
            for k in range(kheight):
                for l in range(kwidth):
                    total = (total + ker[k][l]*imgin[x-hf+k][j-wf+l])
            imgconv[x][j] = total
            
    return imgconv

def xDog(imgin, r1, r2):
    
    filter1 = gfiltergen(r1)
    img1 = actconv(imgin, filter1)
    filter2 = gfiltergen(r2)
    img2 = actconv(imgin, filter2)
    imgfin = img1 - img2
    return imgfin

"TEST IMG"

img = xDog(candy, 3, 6)    


img = cv2.imread('noise_impulsive.png', cv2.IMREAD_GRAYSCALE)
img_out = img.copy()



"Median Filter"

height = img.shape[0]
width = img.shape[1]

for i in np.arange(3, height-3):
    for j in np.arange(3, width-3):
        neighbors = []
        for k in np.arange(-3, 4):
            for l in np.arange(-3, 4):
                a = img.item(i+k, j+l)
                neighbors.append(a)
        neighbors.sort()
        median = neighbors[24]
        b = median
        img_out.itemset((i,j), b)

cv2.imwrite('saltpepper.jpg', img_out)



"Edge Detector"


def magnitude(img1, img2):
    imgcopy = np.zeros(img1.shape)
    ang = np.arctan(img2/img1)
    ang = ang%180
    for row in ang:
        for cell in row:
            if math.isnan(cell.any()):
                cell = 0
            else:
                cell = cell
                
            
    print(ang)
    for x in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            boundry = (img1[x][j]**2 + img2[x][j]**2)**(1/2)
            
            if((boundry>90).all()):
                imgcopy[x][j] = 255
            else:
                imgcopy[x][j] = 0
                
    return imgcopy

"SOBEL OPERATOR"
xgar = np.zeros(shape=(3,3))
xgar[0,0] = -1
xgar[0,1] = -2
xgar[0,2] = -1
xgar[1,0] = 0
xgar[1,1] = 0
xgar[1,2] = 0
xgar[2,0] = 1
xgar[2,1] = 2
xgar[2,2] = 1

ygar = np.zeros(shape=(3,3))
ygar[0,0] = -1
ygar[0,1] = 0
ygar[0,2] = 1
ygar[1,0] = -1
ygar[1,1] = 0
ygar[1,2] = 1
ygar[2,0] = -2
ygar[2,1] = 0
ygar[2,2] = 2


gradx = actconv(img, xgar)
grady = actconv(img, ygar)
edged = magnitude(gradx, grady)
cv2.imwrite("edged.png", edged)


cartoon = img_out.copy()
height2 = cartoon.shape[0]
width2 = cartoon.shape[1]
for x in range(height2):
        for j in range(width2):
            cartoon[x][j] = cartoon[x][j] + edged[x][j]
cv2.imwrite("cartoon.png", cartoon)







        