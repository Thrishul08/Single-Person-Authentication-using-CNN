# Single Person Authentication using CNN 

## Description:

This project implements a basic face authentication system using CNN that detects faces and authenticate users by comparing captured face features with stored data without using any pre-trained model

## Main technologies Used:
Python opencv <br>
tensorflow <br>
numpy <br>
pymongo <br>
mongodb <br>

## Files:

capture.py - code for capturing images through webcam and storing them in the mongodb database - FaceDB <br>
store_dataset.py - code for storing the kaggle dataset from folders into the mongodb database <br>
train_model.py - code for training the model with the images stored in the mongodb database <br>
athenticate.py - testing the model and authentication of the user <br>
face_cnn_model.h5 - trained model <br>
requirements.txt - required libraries for the project <br>

## Progress:

<b>Initial Commit:</b><br>
Used cosine similarity for comparing the features. <br>
<i>problem:</i> focusing on the background features as well instead of just face.


