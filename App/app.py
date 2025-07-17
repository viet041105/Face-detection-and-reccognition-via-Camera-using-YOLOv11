import os
import cv2
import pickle
import time
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.encoders import jsonable_encoder
import uvicorn
from ultralytics import YOLO


# Import các hàm xử lý từ các module của bạn
from detection import model
from recognition import process_frame, save_best_face_images, verify_attendance
from gallery_module import load_gallery_facenet, GALLERY_DIR, gallery_file


# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


app = FastAPI()


def process_video(video_path: str):
    logging.info(f"Bắt đầu xử lý video: {video_path}")
   
    gallery_path = r"C:\Users\baoph\Downloads\App2\gallery_facenetID.pkl"
    yolo_model_path = r"C:\Users\baoph\Downloads\Attendance_AI\face_best.pt"
    output_dir = r"C:\Users\baoph\Downloads\App2\App\best_faces"


    with open(gallery_path, "rb") as f:
        gallery = pickle.load(f)
    logging.info("Gallery loaded successfully.")


    yolo_model = YOLO(yolo_model_path)
    logging.info("YOLO model loaded successfully.")


    best_face_records = {}  
    detection_history = {}  
    unknown_detected = [False]  # Khởi tạo biến theo dõi unknown


    start_time = time.time()
    tracking_duration = 200  


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Cannot open video!")
        raise Exception("Cannot open video!")


    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break


        frame_count += 1
        if frame_count % 10 == 0:
            logging.info(f"Đã xử lý {frame_count} frames...")


        process_frame(frame, yolo_model, gallery, best_face_records, detection_history, unknown_detected)


        elapsed_time = time.time() - start_time
        if elapsed_time >= tracking_duration:
            break


    cap.release()
    logging.info("Hoàn thành xử lý video.")
   
    save_best_face_images(best_face_records, output_dir)
    logging.info("Best face images saved successfully.")
   
    final_attendance = verify_attendance(detection_history, gallery, similarity_threshold=0.65, vote_threshold=0.7)
    logging.info("Final attendance verification completed.")


    # Thêm cảnh báo vào response nếu có unknown
    if unknown_detected[0]:
        logging.info("Cảnh báo: Có người không xác định trong video.")
        return {
            "final_attendance": final_attendance,
            "warning": "Có người không xác định trong video."
        }


    return final_attendance


@app.post("/uploadvideo")
async def upload_video(video: UploadFile = File(...)):
    if video is None or video.filename == "":
        logging.error("Không có file nào được tải lên.")
        return {"error": "No file uploaded."}
   
    logging.info(f"Đã nhận file: {video.filename}")
   
    temp_dir = "temp_videos"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, video.filename)


    # Lưu file tạm
    with open(file_path, "wb") as f:
        f.write(await video.read())
    logging.info(f"File đã lưu tạm thời tại: {file_path}")


    try:
        final_attendance = process_video(file_path)
        print(f"Final return structure: {type(final_attendance)}")
        print(f"Final return data (to be sent in response): {final_attendance}")
       
        response = {
            "message": "Video processed successfully.",
            "final_attendance": final_attendance
        }
        print(f"Complete response object: {response}")
       
        logging.info("Video processed successfully.")
    except Exception as e:
        logging.error(f"Lỗi xử lý video: {str(e)}")
        response = {"error": str(e)}
    finally:
        os.remove(file_path)
        logging.info("File video tạm đã bị xóa.")
   
    return jsonable_encoder(response)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

