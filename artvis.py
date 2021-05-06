# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:40:11 2019

@author: mhamz
"""

"LIBRARIES USED"

import numpy as np
import math
from PIL import Image
import cv2 
from skimage import data
import matplotlib.pyplot as plt
hiddenimports=['pywt._extensions._cwt']

"IMAGE INITILIZATIONS"

monkey = cv2.imread('lemonke.png', cv2.IMREAD_GRAYSCALE)
plane = cv2.imread('plane.png', cv2.IMREAD_GRAYSCALE)
jet = cv2.imread('jet.png', cv2.IMREAD_GRAYSCALE)
moe = cv2.imread('moe.jpg', cv2.IMREAD_GRAYSCALE)
earth = cv2.imread('earth.jpg')
candy = cv2.imread('candy.jpg')

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

"MEDIAN FILTER"

def medfilter(img):
    
    height = img.shape[0]
    width = img.shape[1]
    imgout = img.copy()

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
            imgout.itemset((i,j), b)
    
    return imgout

"EDGE DETECTION"

def converter(imgin):
    
    imgin2 = imgin.copy()
 
    for x in range(imgin.shape[0]):
        for j in range(imgin.shape[1]):
            imgin2[x][j] = imgin[imgin.shape[0]-x-1][imgin.shape[1]-j-1]
    print(imgin2)
    return imgin2


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
    imgcopy = np.uint8(imgcopy)
    for x in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            boundry = (img1[x][j]**2 + img2[x][j]**2)**(1/2)
                
            if((boundry>90).all()):
                imgcopy[x][j] = 255
            else:
                imgcopy[x][j] = 0
          
    return imgcopy

"TEST FOR JET"

jetx = actconv(jet, xgar)
jety = actconv(jet, ygar)
jete = magnitude(jetx, jety)
cv2.imwrite("jete.png", jete)

jetmed = medfilter(jet)
jetcartoon = jet.copy()
heighjet = jet.shape[0]
widthjet = jet.shape[1]
for x in range(heighjet):
        for j in range(widthjet):
            jetcartoon[x][j] = jetmed[x][j] + jete[x][j]
            if jetcartoon[x][j] > 255:
                jetcartoon[x][j] = 0
            else:
                jetcartoon[x][j] = jetcartoon[x][j]
            
cv2.imwrite("jetcartoon.png", jetcartoon)


jetx = actconv(jet, xgar)
jety = actconv(jet, ygar)
jete = magnitude(jetx, jety)
cv2.imwrite("jete.png", jete)

jetmed = medfilter(jet)
jetcartoon = jet.copy()
heighjet = jet.shape[0]
widthjet = jet.shape[1]
for x in range(heighjet):
        for j in range(widthjet):
            jetcartoon[x][j] = jetmed[x][j] + jete[x][j]
            if jetcartoon[x][j] > 255:
                jetcartoon[x][j] = 0
            else:
                jetcartoon[x][j] = jetcartoon[x][j]
            
cv2.imwrite("jetcartoon.png", jetcartoon)


moex = actconv(moe, xgar)
moey = actconv(moe, ygar)
moee = magnitude(moex, moey)
cv2.imwrite("moee.png", moee)

moemed = medfilter(moe)
moecartoon = moe.copy()
heighmoe = moe.shape[0]
widthmoe = moe.shape[1]
for x in range(heighmoe):
        for j in range(widthmoe):
            moecartoon[x][j] = moemed[x][j] + moee[x][j]
            if moecartoon[x][j] > 255:
                moecartoon[x][j] = 0
            else:
                moecartoon[x][j] = moecartoon[x][j]



            
cv2.imwrite("moecartoon.png", moecartoon)




