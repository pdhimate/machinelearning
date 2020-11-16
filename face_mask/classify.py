import os
from shutil import copy

import numpy as np
import cv2
import imutils

from keras.preprocessing import image
from keras.models import load_model

from tqdm import tqdm

#--------------------------------------Loading Files---------------------------------#

# Loading face mask recognition model
model = load_model('C:/Users/Jaspreet Singh/Downloads/model.h5') #load model
       
#------------------------------------Directories---------------------------------#
        
inputdir = 'C:/Users/Jaspreet Singh/Documents/face_mask_detection/face_mask/dataset/raw/NewImages'
outputdir_with_mask = 'C:/Users/Jaspreet Singh/Documents/face_mask_detection/face_mask/dataset/raw/with_mask'
outputdir_without_mask = 'C:/Users/Jaspreet Singh/Documents/face_mask_detection/face_mask/dataset/raw/without_mask'

# Cascade for face detection
face_cascade = cv2.CascadeClassifier('C:/Users/Jaspreet Singh/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')

# List for possible outcomes
mask = ('Mask', 'No Mask')

# For every file in the directory
for file in tqdm(os.listdir(inputdir)):
    
    file_name = os.path.join(inputdir,file)
    #print(file)

    # Reading file as image
    img = cv2.imread(file_name)

    # Detecting faces
    faces = face_cascade.detectMultiScale(img, 1.1, 4)

    # If face detected then crop the face part
    if len(faces) > 0:
        #print(len(faces))
        for (x,y,w,h) in faces:
            #print(faces)
            face = img[int(y):int(y+h), int(x):int(x+w)]
        img = face

    try:
        #cv2.imshow('Mask Detector', img)
                
        # Resizing the image
        img = cv2.resize(img, (64, 64))

        # Converting image to pixels
        img_pixels = image.img_to_array(img)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
            
        # Scalling image
        img_pixels = img_pixels/255

        # Using the model to predict 
        predictions = model.predict(img_pixels)
        
        #print(predictions)
            
        # Finding the index with most value
        max_index = np.argmax(predictions[0])
            
        # Finding corresponding result from list
        mask_detect = mask[max_index]

        print(mask_detect)

        # If the prediction is mask copy image into with_mask folder else.....
        if(mask_detect=='Mask'):
            copy(file_name, outputdir_with_mask)
        else:
            copy(file_name, outputdir_without_mask)
            
    except:
        print("an error occured")
        print("\n"+file)
        
    # Closing OpevCV		
    cv2.waitKey(0)  
      
    #closing all open windows  
    cv2.destroyAllWindows()
