import face_recognition
import cv2
import numpy as np
import glob
import os
import logging
import matplotlib.pyplot as plt
from progress.bar import Bar
import pickle


def face_embeddings(img) :

    faces = face_recognition.face_locations(img) #detect location of faces in the image

    face_encodings = face_recognition.face_encodings(img , faces) #get the face encodings for the faces

    return faces , face_encodings





test_filename = '/home/mactavish/Computer Vision Projects/Face Recognition/1554745456_150874_noticia_normal.jpg' #test file to recognise faces(can have multiple faces)
test_img = cv2.imread(test_filename)

faces , face_encodings = face_embeddings(test_img) #get face locations and embeddings for test image

distance_max = 0.6 


with open('Faces.pickle', 'rb') as handle:
    face_recognition_database = pickle.load(handle)

# print(face_recognition_database.keys())



#acccessing values of the dictionary and cleaning up the data a bit to get rid of extra braces and words like array
ground_truth_face_encodings = str([item1 for item1 in face_recognition_database.values()])
ground_truth_face_encodings = np.array(ground_truth_face_encodings.replace('[','').replace(']','').replace('array','').replace('(','').replace(')','').split(','),dtype=float).reshape(-1,128)


for face , encoding in zip(faces , face_encodings) :
    distances = face_recognition.face_distance(ground_truth_face_encodings , encoding) #computing the difference/distance between the ground truth encoding an test image encoding
    predicted_name = 'Unknown' 
    if np.any(distances <= distance_max) :
        idx = np.argmin(distances) #get index of minimum distance
        #now that we know index, we can easily find the key for this value,i.e., name of the person
        #in case the key isn't found, predicted_name = 'Unknown'
        for keys1, values1 in face_recognition_database.items() :
            values1 = str(values1)
            values1 = np.array(values1.replace('[','').replace(']','').replace('array','').replace('(','').replace(')','').split(','),dtype=float).reshape(-1,128)
            if np.any(values1 == ground_truth_face_encodings[idx]) :
                predicted_name = keys1
    y1 , x2 , y2 , x1 = face #get top, right , bottom and left coordinates of the face
    color = (0,255,0)
    cv2.rectangle(test_img , (x1 , y1) , (x2 , y2) , color , 2) #draw a rectangle around the face using the coordinates
    top_left = (face[3], face[2])
    bottom_right = (face[1], face[2] + 40)

    # cv2.putText(test_img , predicted_name , (x1-20,y1-20) , cv2.FONT_ITALIC , 2 , (0,255,255) , thickness=2) #put text of the person identified
    cv2.rectangle(test_img, top_left, bottom_right, color, cv2.FILLED)
    cv2.putText(test_img, predicted_name, (face[3], face[2]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    print(predicted_name)

cv2.imshow('Image' , test_img)
if(cv2.waitKey(0)&0xFF == 27) :
    cv2.destroyAllWindows()