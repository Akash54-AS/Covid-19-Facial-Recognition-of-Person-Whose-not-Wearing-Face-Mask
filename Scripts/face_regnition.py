# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 23:52:42 2020

@author: Admin
"""
import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from os import listdir
import face_recognition

#-----------------------

face_cascade = cv2.CascadeClassifier(r'C:\Users\Admin\Desktop\Machine Learning\face_detector\model\haarcascade_frontalface_default.xml')



#put your employee pictures in this path as name_of_employee.jpg
employee_pictures = r'C:\Users\Admin\Desktop\Machine Learning\face_detector\dataset'

employees = []
names=[]
for file in listdir(employee_pictures):
	employee, extension = file.split(".")
	img = face_recognition.load_image_file(r'C:\Users\Admin\Desktop\Machine Learning\face_detector\dataset\%s.jpeg' % (employee))
	embeddings =face_recognition.face_encodings(img)[0] 
	names.append(employee)
	employees.append(embeddings)
	
print("employee representations retrieved successfully")

#------------------------
# Initialize some variables
face_locations = []
face_encodings = []
process_this_frame = True

cap = cv2.VideoCapture(0) #webcam

while True:
 # Grab a single frame of video
    ret, frame = cap.read()
    
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    name = "Unknown"     
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(employees, face_encoding)
    
        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = names[first_match_index]
    
    process_this_frame = not process_this_frame
    
    print(name)    
    # Display the results
    print(face_locations)
    for i in face_locations:
        (top,right,bottom,left)=i
        top*=4
        left*=4
        right*=4
        bottom*=4
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 2), 2) 
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (67,67,67), 2) 
        
    cv2.imshow('img',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break	
#kill open cv things		
cap.release()
cv2.destroyAllWindows()