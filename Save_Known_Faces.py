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


def database(path) :

    image_database = {} #dictionary where keys are the name of the person and values are the embeddings of the faces of that person in multiple pictures
    bar = Bar('Processing' , max = len([name for name in os.listdir(path)]))
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
        bar.next()
    bar.finish()
    return image_database




img_database = '/home/mactavish/Computer Vision Projects/Face Recognition/Database' #folder of images from which we acquire the ground truth face embeddings
#each image in the folder above mentioned should have only the face of the person as mentioned in filename, doesn't have to be cropped to the face though
distance_max = 0.6 #threshold for matching faces, higher it is, the more lenient it is, lower value implies it is stricter
face_recognition_database = database(img_database) #get database


with open('Faces.pickle', 'wb') as handle:
    pickle.dump(face_recognition_database, handle, protocol=pickle.HIGHEST_PROTOCOL) #dump the saved embeddings along with the names to a pickle file, which we can load for testing on unknown faces