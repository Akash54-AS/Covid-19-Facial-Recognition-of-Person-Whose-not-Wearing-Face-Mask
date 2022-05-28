# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 23:09:52 2020

@author: Admin
"""
import cv2
from mtcnn import MTCNN
import os

imagepath=r'C:\Users\Admin\Desktop\b.jpeg'
img= cv2.imread(imagepath)
detector=MTCNN()
faces=detector.detect_faces(img)
directory=r'C:\Users\Admin\Desktop'
os.chdir(directory)
for face in faces :
    x1,y1,w,h=face['box']
    x2=x1+w
    y2=y1+h
    (x1,y1) = (max(0,x1), max(0,y1))
    (x2,y2) = (min(img.shape[1]-1,x2), min(img.shape[0]-1,y2))
    image=img[y1:y2,x1:x2]
cv2.imwrite('Babaji.jpeg',image)