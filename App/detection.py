# from ultralytics import YOLO
# import cv2
# # yolo task=detect mode=train data=face-detection.yaml model=yolov8n.pt epochs=50 device=0

# model = YOLO("D:/AI/AiAttendanceProject/App/runs/detect/train/weights/best.pt")

from ultralytics import YOLO
import cv2
import os


# Load model
model = YOLO(r"C:\Users\baoph\Downloads\App\runs\detect\train\weights\best.pt")

