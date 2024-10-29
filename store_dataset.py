# code for storing the dataset into the database

import os
import cv2
import pymongo
from bson import Binary
import gridfs

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["FaceDB"]
collection = db["images"]
fs = gridfs.GridFS(db)

def store_dataset_images(dataset_dir):
    for username in os.listdir(dataset_dir):
        user_dir = os.path.join(dataset_dir, username)
        if os.path.isdir(user_dir):  # Check if it's a directory
            print(f"Storing images for user: {username}")
            
            for image_file in os.listdir(user_dir):
                image_path = os.path.join(user_dir, image_file)
                
                # Load the image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load {image_path}")
                    continue
                
                # Encode image as PNG and convert to binary
                _, img_encoded = cv2.imencode('.png', image)
                img_bytes = Binary(img_encoded.tobytes())
                
                # Store in MongoDB
                collection.insert_one({
                    "username": username,
                    "image_data": img_bytes,
                    "image_name": image_file
                })
                print(f"Stored {image_file} for {username}")

# Usage: Call the function with the path to your dataset
store_dataset_images("../Extracted faces/Extracted Faces")
