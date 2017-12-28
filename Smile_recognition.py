#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: igor
"""
#Imports OpenCV
import cv2

#Loading the cascades used to recognize the haar features
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smileCascade = cv2.CascadeClassifier("haarcascade_smile.xml")

#This function detects the features
#Arguments:
#	gray: The converted gray image
#	frame: Original image
#Return:
#	frame: The original image with the rectagles drawn around features detected
def detect(gray, frame):
    #Detects multiple faces with the cascade
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
	#For each face, a rectangle will be drawn around it and the region of the face
	#will be the region of interest for the other features
        cv2.rectangle(frame, (x,y) , (x+w,y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

	#Detects multiple eyes with the cascade
        eyes = eyeCascade.detectMultiScale(roi_gray, 1.1, 22)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey) , (ex+ew,ey+eh), (0,255,0), 2)

	#Detects multiple smiles with the cascade
        smiles = smileCascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color, (sx,sy) , (sx+sw,sy+sh), (0,0,255), 2)

    return frame

#Turn on the webcam to capoture the video
videoCapture = cv2.VideoCapture(0)

while True:
    #Select the last frame available
    _,frame = videoCapture.read()
    #Turn the frame into gray scale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Call the function to detect the features
    canvas = detect(gray,frame)
    #Shows the frame in a window
    cv2.imshow('Video',canvas)

    #Press 'q' to stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Stop the webcam and close the video window
videoCapture.release()
cv2.destroyAllWindows() 
