import numpy as np
import cv2
import pymongo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from bson.binary import Binary

# MongoDB client setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["FaceDB"]
collection = db["images"]

IMAGE_SIZE = (128, 128)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_face(image):
    """Detects a face in an image and returns the cropped face area if found."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, IMAGE_SIZE)
        return face
    
    return None  # Return None if no face is detected

def load_images_from_mongodb():
    X = []
    y = []
    label_map = {}
    current_label = 0

    for user in collection.distinct("username"):
        images = collection.find({"username": user})

        if user not in label_map:
            label_map[current_label] = user

        for img_data in images:
            img_bytes = np.frombuffer(img_data['image_data'], dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

            # Detect and crop the face using Haar Cascade
            face = detect_and_crop_face(img)
            if face is not None:
                X.append(face)
                y.append(current_label)

        current_label += 1

    X = np.array(X)
    y = np.array(y)
    print(f"Loaded {len(X)} images from {len(label_map)} users.")
    return X, y, label_map

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_model():
    X, y, label_map = load_images_from_mongodb()

    # Normalize images and one-hot encode labels
    X = X.astype('float32') / 255.0  
    y = to_categorical(y, num_classes=len(label_map))  

    # Split dataset for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and compile the model
    model = build_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=len(label_map))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save the trained model
    model.save('face_cnn_model.h5')
    print("Model saved as face_cnn_model.h5")

if __name__ == "__main__":
    train_model()
