import os
import cv2
import pickle
import numpy as np
from fastapi import FastAPI, UploadFile, File
import uvicorn
from ultralytics import YOLO
import time

# Import internal modules for detection and recognition
from detection import model
from recognition import recognize_face_facenet, process_frame, save_best_face_images, verify_attendance
from gallery_module import load_gallery_facenet, GALLERY_DIR, gallery_file


# Main function for local video processing and face recognition
def main():
    # Path to the face gallery (.pkl file with precomputed embeddings)
    gallery_path = r"D:/AI/AiAttendanceProject/Data/gallery_facenet1.pkl"

    # Path to the video file to process
    video_path = r"D:/download/7mins.mp4"

    # Path to YOLOv8 model weights
    yolo_model_path = r"C:\Users\baoph\Documents\Engage\face_best.pt"

    # Directory where best face images will be saved
    output_dir = r"D:/AI/AiAttendanceProject/Data/best_faces"

    # Load gallery (name -> list of embeddings)
    with open(gallery_path, "rb") as f:
        gallery = pickle.load(f)

    # Load YOLOv8 model for face detection
    yolo_model = YOLO(yolo_model_path)

    # Initialize structures for storing results
    best_face_records = {}      # name -> best face info (distance, image, embedding)
    detection_history = {}      # name -> list of all embeddings detected
    unknown_detected = [False]  # Flag to track if any unknown face was found

    # Set time control for video processing
    start_time = time.time()
    tracking_duration = 420  # Process for 7 minutes (420 seconds)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video!")
        return

    # Process the video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Process the current frame: detect, recognize, update logs
        output_frame = process_frame(frame, yolo_model, gallery, best_face_records, detection_history, unknown_detected)

        # Overlay remaining processing time on the video
        elapsed_time = time.time() - start_time
        remaining_time = max(0, tracking_duration - elapsed_time)
        time_text = f"Remaining time: {int(remaining_time)} seconds"
        cv2.putText(output_frame, time_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the annotated frame in a window
        cv2.imshow("Face Recognition", output_frame)

        # Press 'q' to manually exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Stop if time is up
        if elapsed_time >= tracking_duration:
            break

    # Release video resources and close window
    cap.release()
    cv2.destroyAllWindows()

    # Warn if any unknown person was detected
    if unknown_detected[0]:
        print("⚠️ Warning: Unknown person detected in the video.")

    # Save best face images for each recognized person
    saved_paths = save_best_face_images(best_face_records, output_dir)

    # Print raw recognition results
    print("\n--- Recognition results (raw) ---")
    print(f"Total recognized persons: {len(best_face_records)}")
    sorted_persons = sorted(best_face_records.items(), key=lambda x: x[1]['distance'])
    for name, data in sorted_persons:
        print(f"Name: {name}, Best Distance: {data['distance']:.4f}, Saved at: {saved_paths.get(name, 'Not saved')}")

    # Final voting-based attendance verification
    final_attendance = verify_attendance(
        detection_history,
        gallery,
        similarity_threshold=0.65,  # Minimum cosine similarity to consider valid
        vote_threshold=0.7          # At least 70% of detected embeddings must match
    )

    # Print confirmed attendance results
    print("\n--- Final Attendance ---")
    for name, vote_ratio in final_attendance.items():
        print(f"{name} confirmed with vote ratio: {vote_ratio:.2f}")

    print(f"\n✅ Best face images saved to: {output_dir}")

    # Return results for further use if needed
    return best_face_records, final_attendance


# Entry point for running the script directly
if __name__ == "__main__":
    main()
