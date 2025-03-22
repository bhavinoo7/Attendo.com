from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import cv2
import shutil
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from deepface import DeepFace
import mediapipe as mp
from scipy.spatial.distance import cosine
import os
import requests
import tempfile
import subprocess
import time
from pydantic import BaseModel
from numpy.linalg import norm

# FFmpeg path (change if needed)
FFMPEG_PATH = r"C:\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"  

# Disable unnecessary TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Directories and Database Setup
DATASET_DIR = "videos/"
os.makedirs(DATASET_DIR, exist_ok=True)
ENCODINGS_FILE = "face_encodings.pkl"
MONGO_URI = "mongodb+srv://bhavin:bhavin@cluster0.cs8ai.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client["attendance_system"]

# Mediapipe FaceMesh for face alignment validation
mp_face_mesh = mp.solutions.face_mesh

def analyze_face(video_path, position):
    """Validates if the face is correctly positioned (center, left, right)."""
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

# Upload and verify face position
@app.post("/analyze/{position}")
async def upload_video(position: str, video: UploadFile = File(...)):
    file_location = f"{DATASET_DIR}/{position}.webm"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    if analyze_face(file_location, position):
        return {"success": True}
    else:
        return {"success": False, "message": f"Face not properly aligned for {position}"}

# Train face recognition model
@app.post("/train")
async def train_model(student_id: str = Form(...), student_name: str = Form(...)):
    face_encodings = {}

    print(f"Training for: {student_id}, {student_name}")

    for position in ["center", "left", "right"]:
        video_path = f"{DATASET_DIR}/{position}.webm"
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Warning: No frame found in {position} video")
            continue

        img_path = f"{DATASET_DIR}/temp_{position}.jpg"
        cv2.imwrite(img_path, frame)

        try:
            embedding = DeepFace.represent(img_path, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
            face_encodings[position] = embedding
        except Exception as e:
            print(f"Error extracting embedding for {position}: {e}")

    if face_encodings:
        db.student.insert_one({
            "student_id": student_id,
            "student_name": student_name,
            "face_encodings": face_encodings
        })
        return {"message": "Training completed!"}
    else:
        return {"message": "No valid face embeddings, training failed."}

# Define VideoRequest model
class VideoRequest(BaseModel):
    video_url: str

# Download Video from URL
def download_video(video_url, save_path="video.mkv"):
    """Downloads a video file from the given URL."""
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"✅ Video downloaded to {save_path}")
        return save_path
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error downloading video: {str(e)}")

# Convert MKV to MP4 using FFmpeg
def convert_mkv_to_mp4(input_path, output_path="."):
    """Converts MKV to MP4 using FFmpeg."""
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_path, f"{base_name}_{timestamp}.mp4")

        subprocess.run([FFMPEG_PATH, "-i", input_path, "-vcodec", "libx264", "-acodec", "aac", output_path], check=True)
        print(f"✅ Video converted to MP4: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error converting video: {str(e)}")

# Extract Frames from Video (1 Frame per Second)
def extract_frames_per_second(video_path, output_folder="frames", frame_rate=1):
    """Extracts one frame per second from the video."""
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Error opening video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise HTTPException(status_code=500, detail="FPS is zero, cannot extract frames")
    fps = int(fps)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted_frames = []
    frame_count = 0

    # Continue while our calculated frame index is within the video
    while (frame_count * frame_rate * fps) < total_frames:
        timestamp = frame_count * frame_rate
        # Calculate the frame index based on the timestamp and FPS
        frame_index = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret:
            break

        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        extracted_frames.append(frame_path)
        frame_count += 1

    cap.release()
    return extracted_frames


# Extract Face Embedding from an Image
def get_face_embedding(image_path):
    """Extracts face embedding using the ArcFace model."""
    try:
        embeddings = DeepFace.represent(image_path, model_name="ArcFace", enforce_detection=False)
        if embeddings:
            return embeddings[0]["embedding"]
        return None
    except Exception:
        return None

# Recognize Faces More Accurately by Aggregating Votes over Frames
def recognize_face_from_frames(frame_paths):
    """Recognizes students from extracted frames by aggregating multiple detections."""
    students = list(db.student.find({}, {"student_id": 1, "face_encodings": 1}))
    if not students:
        return {"message": "No students found in database", "recognized_students": []}

    # For stricter matching, lower threshold and require multiple votes
    min_match_threshold = 0.3  # STRONGER MATCHING: Lower than before
    min_votes_required = 3     # Must see the same student in at least 3 frames

    # Dictionary to collect votes and distances for each student
    votes = {}  # student_id -> list of distances
    for student in students:
        votes[student["student_id"]] = []

    # Process each frame and accumulate votes
    for frame_path in frame_paths:
        embedding = get_face_embedding(frame_path)
        if embedding is None:
            continue

        best_match_student = None
        best_distance = 1.0  # Initialize with a high distance

        # Compare current frame's embedding against each student's embeddings
        for student in students:
            student_id = student["student_id"]
            stored_encodings = student["face_encodings"]

            for stored_embedding in stored_encodings.values():
                distance = cosine(np.array(embedding), np.array(stored_embedding))
                if distance < best_distance:
                    best_distance = distance
                    best_match_student = student_id

        # Only consider a match if the best distance is strictly below our threshold
        if best_match_student and best_distance < min_match_threshold:
            votes[best_match_student].append(best_distance)

    # Aggregate results by ensuring a student has enough votes and a good median distance
    recognized_students = []
    for student_id, distance_list in votes.items():
        if len(distance_list) >= min_votes_required:
            median_distance = np.median(distance_list)
            if median_distance < min_match_threshold:
                confidence = round(1 - median_distance, 2)  # Higher confidence for lower distances
                recognized_students.append({
                    "student_id": student_id,
                    "votes": len(distance_list),
                    "confidence": confidence
                })

    if not recognized_students:
        return {"message": "Face not recognized", "recognized_students": []}

    return {"message": "Face recognition completed", "recognized_students": recognized_students}

# Recognize Faces from Video
@app.post("/recognize")
async def recognize_face_from_video(video_request: VideoRequest):
    """Receives a video URL, extracts frames, and recognizes faces."""
    video_url = video_request.video_url
    if not video_url:
        raise HTTPException(status_code=400, detail="Video URL is required")

    # Download and convert video if necessary
    video_path = download_video(video_url)
    if video_path.endswith(".mkv"):
        video_path = convert_mkv_to_mp4(video_path)

    # Extract one frame per second from the video
    frame_paths = extract_frames_per_second(video_path, output_folder="frames", frame_rate=1)

    # Recognize faces from the extracted frames
    recognition_result = recognize_face_from_frames(frame_paths)
    print(recognition_result)
    return recognition_result
