# Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED
# 
# This code is made available to the students of 
# the online course titled "OpenCV for Beginners" 
# by Satya Mallick for personal non-commercial use. 
#
# Sharing this code is strictly prohibited without written
# permission from Big Vision LLC. 
#
# For licensing and other inquiries, please email 
# spmallick@bigvisionllc.com 
#
import os
import cv2
import numpy as np

# Initialize HOG parameters
winSize = (64, 128)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
derivAperture = 1
winSigma = -1
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = True
nlevels = 64
signedGradient = False

# OpenCV's HOG based Pedestrian Detector
hogDefault = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                               cellSize, nbins, derivAperture,
                               winSigma, histogramNormType, L2HysThreshold,
                               gammaCorrection, nlevels, signedGradient)
svmDetectorDefault = cv2.HOGDescriptor_getDefaultPeopleDetector()
hogDefault.setSVMDetector(svmDetectorDefault)


# read images from pedestrians directory
imagePath = 'pedestrians/race.jpg'

# We will run pedestrian detector at an fixed height image
finalHeight = 800.0

# read image
im = cv2.imread(imagePath, cv2.IMREAD_COLOR)

# resize image to height finalHeight
scale = finalHeight / im.shape[0]
im = cv2.resize(im, None, fx=scale, fy=scale)

# detectMultiScale using default detector
bboxes, weights = hogDefault.detectMultiScale(im, winStride=(8, 8), padding=(32, 32),
                                                scale=1.05, finalThreshold=2,
                                                hitThreshold=0)

# print pedestrians detected
if len(bboxes) > 0:
  print('Default Detector :: pedestrians detected: {}'.format(bboxes.shape[0]))


for bbox in bboxes:
  x1, y1, w, h = bbox
  x2, y2 = x1 + w, y1 + h
  cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

# Show final result
cv2.imshow("pedestrians", im)
# Write image to disk
cv2.imwrite("result_pedestrian.jpg", im);

cv2.waitKey(0)