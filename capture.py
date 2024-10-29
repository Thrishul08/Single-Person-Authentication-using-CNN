# code for capturing images through webcam and storing them in the mongodb database - FaceDB
# press space for capturing

import cv2
import numpy as np
import pymongo
import gridfs
from bson.binary import Binary
import os


client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["FaceDB"]
collection = db["images"]
fs = gridfs.GridFS(db)

def capture_images(username, num_images=20):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Face")

    print(f"Capturing {num_images} images for user: {username}")
    count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Capture Face", frame)

       
        if cv2.waitKey(1) % 256 == 32:
           
            _, img_encoded = cv2.imencode('.png', frame)
            img_bytes = Binary(img_encoded.tobytes())
            
          
            collection.insert_one({
                "username": username,
                "image_data": img_bytes
            })

            print(f"Image {count + 1} saved to MongoDB for user {username}")
            count += 1

            
            if count >= num_images:
                print("Finished capturing images.")
                break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    username = input("Enter user name: ")
    num_images = int(input("Enter number of images to capture: "))
    capture_images(username, num_images)
