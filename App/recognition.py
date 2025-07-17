import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
import pickle
from ultralytics import YOLO
import time
import os
from datetime import datetime

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained FaceNet model for face embedding
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# Recognize face by comparing against gallery embeddings
def recognize_face_facenet(face_img, gallery, match_threshold=0.25, unknown_threshold=0.35):

    # Convert image to tensor and normalize
    face_tensor = torch.tensor(face_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    face_tensor = torch.nn.functional.interpolate(
        face_tensor, size=(160, 160), mode='bilinear', align_corners=False
    ) / 255.0

    # Get embedding from the FaceNet model
    with torch.no_grad():
        embedding = resnet(face_tensor)
    face_embedding = embedding[0].cpu().numpy()

    best_match = None
    best_distance = float("inf")

    # Compare the face embedding with each stored embedding in the gallery
    for person_name, emb_list in gallery.items():
        for emb in emb_list:
            # Use cosine similarity
            cos_sim = np.dot(face_embedding, emb) / (np.linalg.norm(face_embedding) * np.linalg.norm(emb))
            distance = 1 - cos_sim  # Distance = 1 - similarity

            if distance < best_distance:
                best_distance = distance
                best_match = person_name

    # If the distance is greater than the threshold, mark as unknown
    if best_distance >= unknown_threshold:
        best_match = None

    return best_match, best_distance, face_embedding


# Process a single video frame: detect faces, recognize them, and update records
def process_frame(frame, yolo_model, gallery, best_face_records, detection_history, unknown_detected):
    # Detect faces using YOLO
    results = yolo_model(frame, verbose=False)
    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes in (x1, y1, x2, y2) format

    output = frame.copy()  # Copy of frame for drawing

    for box in xyxy:
        x1, y1, x2, y2 = map(int, box)
        face_crop = frame[y1:y2, x1:x2]  # Crop detected face
        if face_crop.size == 0:
            continue

        # Convert BGR to RGB for face recognition
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Recognize face and get similarity score
        name, distance, face_embedding = recognize_face_facenet(
            face_rgb, gallery, match_threshold=0.25, unknown_threshold=0.35
        )

        if name is None:
            unknown_detected[0] = True  # Flag unknown person found
        else:
            # Update detection history for the recognized person
            if name not in detection_history:
                detection_history[name] = []
            detection_history[name].append(face_embedding)

            # Store best face image (with smallest distance)
            if name not in best_face_records or distance < best_face_records[name]['distance']:
                best_face_records[name] = {
                    'distance': distance,
                    'face_image': face_crop.copy(),
                    'embedding': face_embedding
                }

        # Display results on the output frame
        display_name = name if name is not None else "Unknown"
        color = (0, 255, 0) if name is not None else (0, 0, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{display_name} ({distance:.3f})"
        cv2.putText(output, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return output  # Return annotated frame


# Save the best face image for each recognized person
def save_best_face_images(best_face_records, output_dir='best_faces'):

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use timestamp to make filenames unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths = {}

    # Save each best face image to the output folder
    for name, data in best_face_records.items():
        face_path = os.path.join(output_dir, f"{name}_{timestamp}_{data['distance']:.4f}.jpg")
        cv2.imwrite(face_path, data['face_image'])
        saved_paths[name] = face_path

    return saved_paths  # Return file paths for reference


# Verify attendance by checking how many detected embeddings match gallery
def verify_attendance(detection_history, gallery, similarity_threshold=0.65, vote_threshold=0.5):

    final_attendance = {}

    # Loop through all detected persons and verify
    for person, embeddings_list in detection_history.items():
        gallery_embs = gallery.get(person, [])
        if not gallery_embs:
            continue

        valid_count = 0  # Count how many embeddings pass the similarity threshold

        for det_emb in embeddings_list:
            verified = False
            for gal_emb in gallery_embs:
                # Compute cosine similarity
                cos_sim = np.dot(det_emb, gal_emb) / (np.linalg.norm(det_emb) * np.linalg.norm(gal_emb))
                if cos_sim > similarity_threshold:
                    verified = True
                    break
            if verified:
                valid_count += 1

        # Calculate ratio of verified embeddings
        vote_ratio = valid_count / len(embeddings_list)
        if vote_ratio >= vote_threshold:
            final_attendance[person] = vote_ratio  # Mark as attended

    return final_attendance
