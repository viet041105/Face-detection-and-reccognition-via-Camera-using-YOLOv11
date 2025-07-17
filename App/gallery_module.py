import os
import cv2
import numpy as np
import albumentations as A
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Directory where registered images are stored
GALLERY_DIR = r"D:\AI\AiAttendanceProject\Data\ImageReg"

# Path to save the gallery (embeddings)
gallery_file = r"D:\AI\AiAttendanceProject\Data\gallery_facenet1.pkl"

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)

# Load pre-trained FaceNet model for feature extraction
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# Function to load and process gallery images into embeddings
def load_gallery_facenet(gallery_dir):
    # Define image augmentation pipeline using Albumentations
    transform = A.Compose([
         A.HorizontalFlip(p=0.5),             # Randomly flip image horizontally
         A.Rotate(limit=10, p=0.5),           # Randomly rotate image within ±10 degrees
         A.RandomBrightnessContrast(p=0.5),   # Adjust brightness/contrast
         A.Resize(height=160, width=160)      # Resize image to 160x160
    ])
    
    gallery = {}  # Dictionary to hold person_name -> list of embeddings

    # Loop through each person in the gallery directory
    for person_name in os.listdir(gallery_dir):
        person_path = os.path.join(gallery_dir, person_name)

        # Skip if not a directory
        if not os.path.isdir(person_path):
            continue

        embeddings = []  # Store all embeddings for the current person

        # Loop through each image file for the person
        for filename in os.listdir(person_path):
            img_path = os.path.join(person_path, filename)
            print(f"Processing: {img_path}")

            # Read image using OpenCV
            image = cv2.imread(img_path)
            if image is None:
                print(f"[Warning] Unable to load image {img_path}")
                continue

            # Convert BGR (OpenCV default) to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect and extract face
            face = mtcnn(image_rgb)
            if face is None:
                print(f"[Warning] No face detected in {img_path}")
            else:
                face = face.unsqueeze(0).to(device)  # Add batch dimension
                with torch.no_grad():
                    embedding = resnet(face)  # Generate facial embedding
                embeddings.append(embedding[0].cpu().numpy())  # Store as numpy array

            # Try augmentation and repeat face detection + embedding
            try:
                augmented = transform(image=image_rgb)  # Apply augmentation
                image_aug = augmented["image"]
                face_aug = mtcnn(image_aug)
                if face_aug is not None:
                    face_aug = face_aug.unsqueeze(0).to(device)
                    with torch.no_grad():
                        embedding_aug = resnet(face_aug)
                    embeddings.append(embedding_aug[0].cpu().numpy())
                else:
                    print(f"[Warning] No face detected in augmented image {img_path}")
            except Exception as e:
                print(f"[Warning] Error processing augmented image {img_path}: {e}")

        # If embeddings were successfully generated, store them under the person's name
        if embeddings:
            gallery[person_name] = embeddings

    return gallery  # Return the dictionary containing all embeddings
