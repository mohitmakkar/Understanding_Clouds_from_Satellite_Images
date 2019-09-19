import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
#%%

path_default = os.path.dirname(os.path.realpath(__file__)) + '/'
# path_default = 'C:\git_practice\satellite_cloud\Understanding_Clouds_from_Satellite_Images'
image_folder = path_default + 'sample_train/'
img = cv2.imread(image_folder+'00a0954.jpg')
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

## to view image
#plt.imshow(gray_img)

## resizing the image
smaller_img = cv2.resize(gray_img,(525,350))

## edge detection
#calculate the edges using Canny edge algorithm
edges = cv2.Canny(img,100,200) 
#plot the edges
plt.imshow(edges)
