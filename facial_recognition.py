# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing required libraries
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt
import cv2
from skimage import feature


#reading the images


img1 = imread('1.jpg')
plt.axis("off")
plt.imshow(img1)
print(img1.shape)

#resizing image
resized_img1 = resize(img1, (128*4, 64*4))
plt.axis("off")
plt.imshow(resized_img1)
plt.show()
print(resized_img1.shape)

#creating hog features
fd, hog_image1 = hog(resized_img1, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=True)
print(fd.shape)
print(hog_image1.shape)
plt.axis("off")
plt.imshow(hog_image1, cmap="gray")
plt.show()

# save the images
plt.imsave("resized_img1.jpg", resized_img1)
plt.imsave("hog_image.jpg2", hog_image1, cmap="gray")


img2 = imread('2.jpg')
plt.axis("off")
plt.imshow(img2)
print(img2.shape)

#resizing image
resized_img2 = resize(img2, (128*4, 64*4))
plt.axis("off")
plt.imshow(resized_img2)
plt.show()
print(resized_img2.shape)

#creating hog features
fd, hog_image2 = hog(resized_img2, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=True)
print(fd.shape)
print(hog_image2.shape)
plt.axis("off")
plt.imshow(hog_image2, cmap="gray")
plt.show()

# save the images
plt.imsave("resized_img.jpg2", resized_img1)
plt.imsave("hog_image.jpg2", hog_image2, cmap="gray")



H1 = feature.hog(hog_image1, orientations=9, pixels_per_cell=(8, 8),
		cells_per_block=(2, 2), visualize=True, multichannel=False)
H2 = feature.hog(hog_image1, orientations=9, pixels_per_cell=(8, 8),
		cells_per_block=(2, 2), visualize=True, multichannel= False)

a = cv2.compareHist(H1,H2, cv2.HISTCMP_CORREL)

print(a)