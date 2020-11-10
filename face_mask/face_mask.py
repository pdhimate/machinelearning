import os

import numpy as np
import cv2
import imutils

from keras.preprocessing import image
from keras.models import load_model

#--------------------------------------Loading Files---------------------------------#

# Loading emotion recognition model and weights 
model = load_model('C:/Users/Jaspreet Singh/Downloads/model.h5') #load model
       
#------------------------------------Emotion Detection Part---------------------------------#
        
# Capturing the video using from the path
cap = cv2.VideoCapture(0)

# Emotion labels
mask = ('Mask', 'No Mask')

# Creating empty list which will be used to store emotions
emotion_list = []

# Reading video frame by frame
while(True):
   
    ret, img = cap.read()
    img = imutils.resize(img, width=600)
    
    # Reading till the end of the video
    if ret:

        cv2.imshow('Emotion Recognizer', img)
        
                 
        # Resizing the cropped face
        face = cv2.resize(img, (64, 64))
        
        # Converted face image to pixels
        img_pixels = image.img_to_array(face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
            
        # Scalling image
        img_pixels = img_pixels/255
        
        # Using the model to predict the detected face
        predictions = model.predict(img_pixels)
        print(predictions)
            
            # Finding the index with most value
        max_index = np.argmax(predictions[0])
            
        # Finding corresponding emotion
        mask_detect = mask[max_index]
        
        # Writing detected emotions on the rectangle
        cv2.putText(img, mask_detect, (int(20), int(20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color = (255,255,255),
                    thickness = 1)

          # Showing the frame with detected face
        cv2.imshow('Mask Detector', img)
    
    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Closing OpevCV		
cap.release()
cv2.destroyAllWindows()

#-------------------------------------------------------------------------------------------#
