from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import cv2
import shutil
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import mediapipe as mp
from scipy.spatial.distance import cosine
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU

import requests
import faiss
import tensorflow as tf
from pydantic import BaseModel
from insightface.app import FaceAnalysis
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Fix NumPy deprecation issue
np.float = float  # Monkey patch to avoid deep_sort issues

# Limit GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories and Database Setup
DATASET_DIR = "videos/"
os.makedirs(DATASET_DIR, exist_ok=True)
MONGO_URI = "mongodb+srv://bhavin:bhavin@cluster0.cs8ai.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client["attendance_system"]

# Mediapipe FaceMesh for face alignment validation
mp_face_mesh = mp.solutions.face_mesh

def analyze_face(video_path, position):
    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh()
    face_valid = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                nose_x = landmarks.landmark[1].x * frame.shape[1]
                center_x = frame.shape[1] / 2
                if position == "center" and (center_x - 80 < nose_x < center_x + 80):
                    face_valid = True
                elif position == "left" and nose_x < center_x - 80:
                    face_valid = True
                elif position == "right" and nose_x > center_x + 80:
                    face_valid = True
    cap.release()
    return face_valid

@app.post("/analyze/{position}")
async def upload_video(position: str, video: UploadFile = File(...)):
    file_location = f"{DATASET_DIR}/{position}.webm"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    if analyze_face(file_location, position):
        return {"success": True}
    else:
        return {"success": False, "message": f"Face not properly aligned for {position}"}

@app.post("/train")
async def train_model(student_id: str = Form(...), student_name: str = Form(...)):
    face_encodings = {}
    
    for position in ["center", "left", "right"]:
        video_path = f"{DATASET_DIR}/{position}.webm"
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"❌ No frame extracted from {position} video.")
            continue

        # Detect faces using insightface
        faces = face_recognizer.get(frame)

        if not faces:
            print(f"⚠️ No face detected in {position} video.")
            continue  # Skip if no face found

        # Extract first detected face embedding
        embedding = np.array(faces[0].normed_embedding, dtype=np.float32)
        face_encodings[position] = embedding.tolist()  # Convert NumPy array to list for MongoDB storage

    if not face_encodings:
        raise HTTPException(status_code=400, detail="No valid face embeddings found, training failed.")

    # Insert into MongoDB
    db.student.insert_one({
        "student_id": student_id,
        "student_name": student_name,
        "face_encodings": face_encodings
    })

    return {"message": "Training completed!"}


# Initialize FAISS index
face_dim = 512
faiss_index = faiss.IndexFlatL2(face_dim)
students_db = {}

students = list(db.student.find({}, {"student_id": 1, "face_encodings": 1}))
for student in students:
    student_id = student["student_id"]
    for angle in ["center", "left", "right"]:
        if angle in student["face_encodings"]:
            faiss_index.add(np.array([student["face_encodings"][angle]], dtype=np.float32))
            students_db[len(students_db)] = (student_id, angle)

print("🔍 FAISS Index:", faiss_index.ntotal)

# Initialize models
face_detector = YOLO("models/yolov8n-face.pt")
face_recognizer = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_recognizer.prepare(ctx_id=0, det_size=(640, 640))
tracker = DeepSort(max_age=50)

class VideoRequest(BaseModel):
    video_url: str

def download_video(video_url, save_path="video.mp4"):
    response = requests.get(video_url, stream=True)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    return save_path

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_skip = 5
    frame_count = 0
    results = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            # Ensure the frame is in the correct format (RGB, 3 channels)
            if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
                # Perform face detection using YOLO
                detections = face_detector(frame)

                # Extract bounding boxes and confidence scores
                face_boxes = []
                for detection in detections:
                    if detection.boxes is not None and detection.boxes.xyxy is not None:
                        for box, conf in zip(detection.boxes.xyxy, detection.boxes.conf):
                            box = box.tolist()  # Convert tensor to list
                            conf = conf.item()  # Convert tensor to float
                            if len(box) == 4:  # Ensure it's a valid bounding box
                                x1, y1, x2, y2 = map(int, box)
                                # Format: ([x1, y1, x2, y2], confidence)
                                face_boxes.append(([x1, y1, x2, y2], conf))

                # Debug: Print face_boxes to verify its format
                print("🔍 Face Boxes Detected:", face_boxes)

                # Track faces
                if face_boxes:  # Only proceed if there are valid face boxes
                    try:
                        # Ensure the frame is in the correct format (RGB, 3 channels)
                        if frame.shape[2] != 3:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels if necessary

                        # Pass face_boxes and frame to the tracker
                        tracked_faces = tracker.update_tracks(face_boxes, frame)
                    except Exception as e:
                        print(f"❌ Error in tracker.update_tracks: {e}")
                        continue

                    # Process tracked faces
                    for track in tracked_faces:
                        x, y, w, h = track.to_tlbr()
                        face_crop = frame[int(y):int(h), int(x):int(w)]
                        faces = face_recognizer.get(face_crop)
                        print("👤 Faces Detected:", faces)

                        if faces:
                            for face in faces:  # Process each detected face in the frame
                                embedding = np.array(face.normed_embedding, dtype=np.float32)
                                distances, idx = faiss_index.search(np.array([embedding]), 1)
                                if distances[0][0] < 0.6:  # Confidence threshold
                                    student_data = students_db.get(idx[0][0])
                                    if student_data:
                                        student_id, angle = student_data
                                        print(f"🎓 Student ID: {student_id}, Angle: {angle}")
                                        results[student_id] = results.get(student_id, 0) + 1
                                        print("📝 Attendance:", results)
            else:
                print("⚠️ Invalid frame format. Skipping frame.")

        frame_count += 1
    cap.release()
    return results

@app.post("/recognize")
async def recognize_face_from_video(video_request: VideoRequest):
    video_path = download_video(video_request.video_url)
    recognition_result = process_video(video_path)

    # Transform recognition_result into the desired format
    recognized_students = []
    for student_id, votes in recognition_result.items():
        recognized_students.append({
            "student_id": student_id,
            "votes": votes,
            "confidence": 0.99  # Assuming a fixed confidence value for simplicity
        })

    return {
        "message": "Face recognition completed",
        "recognized_students": recognized_students
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
