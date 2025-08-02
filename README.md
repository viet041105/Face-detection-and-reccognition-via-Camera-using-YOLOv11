# Face-detection-and-reccognition-via-Camera-using-YOLOv11
This project provides an end-to-end AI system that uses YOLOv11-nano (small and fast) for face detection and FaceNet (InceptionResnetV1) for face recognition to perform automated attendance tracking from video or camera input.

It supports both real-time display and API-based video upload for batch processing.

 Features

 Lightweight face detection using YOLOv11n (nano) – optimized for speed and low-resource systems.

 Deep face recognition using FaceNet with cosine similarity.

 Automatically tracks attendance with voting-based validation.

 Identifies unknown individuals not present in the registered gallery.

 Saves the best quality image of each recognized face.

 Displays annotated video frames with live tracking and time countdown.

 Optional API endpoint using FastAPI to upload and analyze videos.

Project Structure bash Copy Edit ├── app.py # FastAPI server (optional) ├── main.py # Local video processing and display ├── detection.py # YOLOv11n model loader ├── recognition.py # Core recognition logic (FaceNet, voting, best face, etc.) ├── gallery_module.py # Gallery loader with data augmentation ├── Data/ │ ├── ImageReg/ # Registered users' images │ ├── gallery_facenet1.pkl # Pickled embeddings of registered users │ └── best_faces/ # Output folder for best face captures ├── temp_videos/ # Temporary uploaded videos (via API) 🛠️ Requirements Install dependencies:

bash Copy Edit pip install -r requirements.txt Key Libraries ultralytics – YOLOv11 face detection (nano model)

facenet-pytorch – Face embedding using InceptionResnetV1

opencv-python – Image and video processing

albumentations – Image augmentation

fastapi, uvicorn – API service (optional)

numpy, torch, pickle, etc.

📽️ How It Works 🔹 1. Register Faces Place images of each person into separate folders under Data/ImageReg/ (e.g. ImageReg/Alice, ImageReg/Bob).

Run gallery_module.py to extract face embeddings and save them to gallery_facenet1.pkl.

🔹 2. Process Video You can process a video locally with live display using:

bash Copy Edit python main.py What happens:

YOLOv11n detects faces every frame.

FaceNet compares them to the gallery.

Best face images are stored.

Final attendance is verified based on vote ratio.

🔹 3. Upload via API (Optional) Run FastAPI server:

bash Copy Edit uvicorn app:app --reload Send a request using curl or Postman:

bash Copy Edit curl -X POST "http://localhost:8000/uploadvideo"
-F "video=@/path/to/video.mp4" ✅ Output Example sql Copy Edit --- Final Attendance --- Alice confirmed with vote ratio: 0.85 Bob confirmed with vote ratio: 0.73

Best face images saved to: Data/best_faces

 Warning: Unknown person detected in the video.

Configuration Parameter Location Description

gallery_file gallery_module.py Path to .pkl gallery yolo_model_path main.py / app.py Path to YOLOv11n weights (.pt) match_threshold recognize_face_facenet() Similarity match for initial filtering unknown_threshold recognize_face_facenet() Threshold to classify unknown face similarity_threshold verify_attendance() Cosine similarity pass threshold vote_threshold verify_attendance() % of embeddings that must match

📸 A picture sample of our detection and recognition from camera ai :
<img width="1264" height="643" alt="image" src="https://github.com/user-attachments/assets/ba3418a3-672e-4e51-a0e8-9d9c18af6e65" />

