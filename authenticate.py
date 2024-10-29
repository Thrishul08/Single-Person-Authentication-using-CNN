# press space to capture

import cv2
import numpy as np
import pymongo
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from bson.binary import Binary

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["FaceDB"]
collection = db["images"]

# Constants
IMAGE_SIZE = (128, 128)
THRESHOLD = 0.7

# Load the pre-trained model
model = load_model('face_cnn_model.h5')

def load_images_from_mongodb():
    X = []
    y = []
    label_map = {}

    for user in collection.distinct("username"):
        images = collection.find({"username": user})

        for img_data in images:
            img_bytes = np.frombuffer(img_data['image_data'], dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            img = cv2.resize(img, IMAGE_SIZE)

            X.append(img)
            y.append(user)

    X = np.array(X)
    y = np.array(y)

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    label_map = {idx: label for idx, label in enumerate(encoder.classes_)}
    
    return X, y_encoded, label_map, encoder

def preprocess_image(image):
    img = cv2.resize(image, IMAGE_SIZE)
    img = img.astype('float32') / 255.0  
    img = np.expand_dims(img, axis=0)
    return img

def capture_face_from_webcam():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Face")

    print("Press SPACE to capture the face")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Capture Face", frame)

        if cv2.waitKey(1) % 256 == 32:  
            print("Captured image from webcam.")
            cam.release()
            cv2.destroyAllWindows()
            return frame

    cam.release()
    cv2.destroyAllWindows()
    return None

def authenticate_user():
    # Load stored images, labels, and mappings
    stored_images, stored_labels, label_map, encoder = load_images_from_mongodb()

    # Capture and preprocess image from webcam
    captured_image = capture_face_from_webcam()
    if captured_image is None:
        print("Failed to capture image from webcam.")
        return

    captured_image_preprocessed = preprocess_image(captured_image)
    captured_features = model.predict(captured_image_preprocessed)

    # Initialize best match variables
    best_match_user = None
    best_match_score = -1  # For cosine similarity, we seek the highest score

    # Compare against stored images using cosine similarity
    for idx, stored_img in enumerate(stored_images):
        stored_img_preprocessed = preprocess_image(stored_img)
        stored_features = model.predict(stored_img_preprocessed)

        # Calculate cosine similarity
        similarity_score = cosine_similarity(captured_features, stored_features)[0][0]

        # Update best match if a higher similarity score is found
        if similarity_score > best_match_score:
            best_match_score = similarity_score
            best_match_user = stored_labels[idx]

    # Check against threshold for authentication
    if best_match_score >= THRESHOLD:
        user_label = encoder.inverse_transform([best_match_user])[0]
        print(f"Authentication successful! User: {user_label}")
    else:
        print("Authentication failed. No matching user found.")

if __name__ == "__main__":
    authenticate_user()
