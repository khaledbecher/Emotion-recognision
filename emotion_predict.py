# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 23:15:34 2023

@author: MAISON INFO
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import model_from_json
import mediapipe
# Load model and weights
json_file = open("emotion.json",'r')
loaded_model_json = json_file.read() 
json_file.close()
model_em = model_from_json(loaded_model_json)
model_em.load_weights("emotion_weights.h5")


# Predict
classes = ["angry","disgust","fear","happy","neutral","sad","surprize"]
cap = cv2.VideoCapture(0)
face_detection = mediapipe.solutions.face_detection
face = face_detection.FaceDetection(min_detection_confidence = 0.5)
while(True):
    ret,frame = cap.read()
    if not ret:
        break;
    frame = cv2.flip(frame,1)
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = face.process(frame_rgb)
     
    if(results.detections):
        for detection in results.detections:
            bBox = detection.location_data.relative_bounding_box
            ih,iw,_ = frame.shape
            x,y,w,h = int(bBox.xmin*iw),int(bBox.ymin*ih),int(bBox.width*iw),int(bBox.height*ih)
            #cv2.rectangle(frame,(x-10,y-80),(x+w+20,y+h),(0,255,0),2)
            cv2.rectangle(frame,(x,y-50),(x+w,y+h+10),(0,255,0),2)
            face_frame = frame[y-100:y+h+40,x-35:x+w+35]
            face_frame = cv2.resize(face_frame,(48,48))
            face_frame_gray = cv2.cvtColor(face_frame,cv2.COLOR_BGR2GRAY)
            face_frame_gray_rescaled = face_frame_gray/255.0
            face_frame_gray_resized = np.expand_dims(np.expand_dims(face_frame_gray_rescaled,axis=0),axis = -1)
            detection = classes[model_em.predict(face_frame_gray_resized).argmax()]
            text_x = x
            text_y = y-110
            cv2.putText(frame,detection,(text_x,text_y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    # # Display the frame
    cv2.imshow("emotion Detection", frame)
    
    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()