import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import pymongo

IMAGE_SIZE = (128, 128)
THRESHOLD = 0.5 

# Load the pre-trained face recognition model
model = load_model('face_cnn_model.h5')

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["FaceDB"]
collection = db["images"]

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_images_from_mongodb():
    X = []
    y = []
    label_map = {}

    for user in collection.distinct("username"):
        images = collection.find({"username": user})
        label_map[user] = user

        for img_data in images:
            img_bytes = np.frombuffer(img_data['image_data'], dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

            # Detect and crop the face
            face = detect_and_crop_face(img)
            if face is not None:
                X.append(face)
                y.append(user)

    return np.array(X), np.array(y), label_map

def detect_and_crop_face(image):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If a face is detected, crop to the face region
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, IMAGE_SIZE)
        return face
    return None

def preprocess_image(image):
    face = detect_and_crop_face(image)
    if face is not None:
        face = face.astype('float32') / 255.0  # Normalize pixel values
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        return face
    else:
        print("No face detected.")
        return None

def capture_face_from_webcam():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Face")

    print("Press SPACE to capture the face image")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Capture Face", frame)

        if cv2.waitKey(1) % 256 == 32:  # SPACE pressed
            print("Captured image from webcam.")
            cam.release()
            cv2.destroyAllWindows()
            return frame

    cam.release()
    cv2.destroyAllWindows()
    return None

def authenticate_user():
    # Load stored images and labels from MongoDB
    stored_images, stored_labels, label_map = load_images_from_mongodb()

    # Capture an image from the webcam
    captured_image = capture_face_from_webcam()
    if captured_image is None:
        print("Failed to capture image from webcam.")
        return

    # Preprocess the captured image
    captured_image_preprocessed = preprocess_image(captured_image)
    if captured_image_preprocessed is None:
        print("No face detected in the captured image.")
        return

    # Extract features from the captured image
    captured_features = model.predict(captured_image_preprocessed)

    best_match_user = None
    best_match_score = float('-inf')

    # Compare with stored images
    for idx, stored_img in enumerate(stored_images):
        stored_img_preprocessed = preprocess_image(stored_img)
        if stored_img_preprocessed is None:
            continue

        stored_features = model.predict(stored_img_preprocessed)
        similarity_score = cosine_similarity(captured_features, stored_features)[0][0]

        if similarity_score > best_match_score:
            best_match_score = similarity_score
            best_match_user = stored_labels[idx]

    # Check if the similarity score passes the threshold
    if best_match_score > THRESHOLD:
        print(f"Authentication successful! User: {best_match_user}")
    else:
        print("Authentication failed. No matching user found.")

# Run authentication
if __name__ == "__main__":
    authenticate_user()
