import face_recognition
import cv2
import numpy as np
import glob
import os
import logging
import matplotlib.pyplot as plt



def face_embeddings(img) :

    faces = face_recognition.face_locations(img) #detect location of faces in the image

    face_encodings = face_recognition.face_encodings(img , faces) #get the face encodings for the faces

    return faces , face_encodings


def database(path) :

    image_database = {} #dictionary where keys are the name of the person and values are the embeddings of the faces of that person in multiple pictures
    for filename in glob.glob(os.path.join(path,'*.jpg')) :
        name1 = []
        image = face_recognition.load_image_file(filename)
        for character in os.path.basename(filename) :
            if character=='_' : #characters before the '_' in the filename is considered as the name of the person
                break
            name1.append(character)
        name = ""
        name = str(name.join(name1))
        #print(name)

        faces , face_encodings = face_embeddings(image) #get the face location and face embeddings
        if faces==[] : #if no face was detected in the picture, next file is loaded
            continue
        else :
            if name in image_database.keys() : #if name is present in the dictionary, face embeddings are simply appended to the values of that key(name)
                image_database[name].append(face_encodings[0])
            else : #if name is not present in dictionary, a new key-value pair is added to the dictionary
                image_database[name]=[face_encodings[0]]
    return image_database




img_database = '/home/mactavish/Computer Vision Projects/Face Recognition/Database' #folder of images from which we acquire the ground truth face embeddings
#each image in the folder above mentioned should have only the face of the person as mentioned in filename, doesn't have to be cropped to the face though
distance_max = 0.6 #threshold for matching faces, higher it is, the more lenient it is, lower value implies it is stricter
face_recognition_database = database(img_database) #get database
test_filename = '/home/mactavish/Computer Vision Projects/Face Recognition/1554745456_150874_noticia_normal.jpg' #test file to recognise faces(can have multiple faces)
test_img = cv2.imread(test_filename)

faces , face_encodings = face_embeddings(test_img) #get face locations and embeddings for test image

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
    cv2.putText(test_img , predicted_name , (x1-20,y1-20) , cv2.FONT_ITALIC , 2 , (0,255,255) , thickness=2) #put text of the person identified
    print(predicted_name)

cv2.imshow('Image' , test_img)
if(cv2.waitKey(0)&0xFF == 27) :
    cv2.destroyAllWindows() 