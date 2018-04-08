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


import dlib,cv2
import numpy as np

def writeLandmarksToFile(landmarks, landmarksFileName):
  with open(landmarksFileName, 'w') as f:
    for p in landmarks.parts():
      f.write("%s %s\n" %(int(p.x),int(p.y)))

  f.close()

def drawLandmarks(im, landmarks):
  for i, part in enumerate(landmarks.parts()):
    px = int(part.x)
    py = int(part.y)
    cv2.circle(im, (px, py), 1, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(im, str(i+1), (px, py), cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 0, 0), 1)

# Landmark model location
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
# Get the face detector
faceDetector = dlib.get_frontal_face_detector()
# The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

# Read image
imageFilename = "girl.jpg"
im= cv2.imread(imageFilename)
# landmarks will be stored in results/family_i.txt
landmarksBasename = "output_python"

# Detect faces in the image
faceRects = faceDetector(im, 0)
print("Number of faces detected: ",len(faceRects))

# List to store landmarks of all detected faces
landmarksAll = []

# Loop over all detected face rectangles
for i in range(0, len(faceRects)):
  newRect = dlib.rectangle(int(faceRects[i].left()),int(faceRects[i].top()),
      int(faceRects[i].right()),int(faceRects[i].bottom()))

  # For every face rectangle, run landmarkDetector
  landmarks = landmarkDetector(im, newRect)
  # Print number of landmarks
  if i==0:
    print("Number of landmarks",len(landmarks.parts()))

  # Store landmarks for current face
  landmarksAll.append(landmarks)
  # Draw landmarks on face
  drawLandmarks(im, landmarks)

  landmarksFileName = landmarksBasename +"_"+ str(i)+ ".txt"
  print("Saving landmarks to", landmarksFileName)
  # Write landmarks to disk
  writeLandmarksToFile(landmarks, landmarksFileName)

outputFileName = "result_python_Landmarks.jpg"
print("Saving output image to", outputFileName)
cv2.imwrite(outputFileName, im)

cv2.imshow("Facial Landmark detector", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

  