import streamlit as st
import cv2
import numpy as np
from PIL import Image
import random
import sqlite3
import datetime
import os
import pandas as pd

st.set_page_config(page_title="RQI — YOLO + Lane + Pothole Mask + GPS", layout="wide")

# =====================================================
# SQLite Database Setup
# =====================================================
conn = sqlite3.connect("potholes.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS pothole_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    class_name TEXT,
    confidence REAL,
    snapshot_path TEXT,
    latitude REAL,
    longitude REAL
)
""")
conn.commit()

# =====================================================
# Snapshots folder
# =====================================================
SNAPSHOT_FOLDER = "snapshots"
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

def insert_detection(class_name, confidence, snapshot_path, latitude=None, longitude=None):
    timestamp = datetime.datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO pothole_events
        (timestamp, class_name, confidence, snapshot_path, latitude, longitude)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (timestamp, class_name, confidence, snapshot_path, latitude, longitude))
    conn.commit()

# =====================================================
# TFLite Interpreter
# =====================================================
Interpreter = None
try:
    from tflite_runtime.interpreter import Interpreter
    st.sidebar.success("Using tflite_runtime")
except:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    st.sidebar.success("Using tensorflow.lite")

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "best_float16.tflite"
CLASS_NAMES = ["Pothole", "Crack", "Faded Lane"]

# =====================================================
# Load YOLO TFLite Model
# =====================================================
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =====================================================
# Lane Detection
# =====================================================
def enhanced_lane_detection(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 40, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([15, 70, 70]), np.array([35, 255, 255]))
    lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

    kernel = np.ones((5, 5), np.uint8)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)

    return cv2.applyColorMap(lane_mask, cv2.COLORMAP_HOT)

# =====================================================
# NEW: Pothole Mask (NO TRAINING)
# =====================================================
def pothole_mask_no_training(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Dark-region detection
    _, mask = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

# =====================================================
# YOLO Utilities
# =====================================================
def generate_colors(n):
    random.seed(42)
    return [tuple(random.randint(0,255) for _ in range(3)) for _ in range(n)]

COLORS = generate_colors(len(CLASS_NAMES))

def preprocess_for_tflite(frame):
    h, w = input_details[0]["shape"][1:3]
    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def run_tflite_inference(frame):
    inp = preprocess_for_tflite(frame)
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])[0]

    if out.size == 0:
        return [], [], []

    return out[:, :4], out[:, 4], out[:, 5]

def draw_boxes(frame, boxes, scores, classes, lat, lon, threshold):
    h, w, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] < threshold:
            continue

        x1, y1, x2, y2 = boxes[i]
        x1, x2 = int(x1*w), int(x2*w)
        y1, y2 = int(y1*h), int(y2*h)

        cid = int(classes[i])
        color = COLORS[cid % len(COLORS)]
        label = f"{CLASS_NAMES[cid]} {scores[i]:.2f}"

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        crop = frame[y1:y2, x1:x2]
        snap = f"{SNAPSHOT_FOLDER}/{CLASS_NAMES[cid]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        if crop.size > 0:
            cv2.imwrite(snap, crop)
            insert_detection(CLASS_NAMES[cid], float(scores[i]), snap, lat, lon)

# =====================================================
# UI
# =====================================================
st.title("RQI — YOLO + Lane + Pothole Mask (Ather-style)")

mode = st.selectbox("Select Mode", ["Upload Image", "Live Webcam / DroidCam"])
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

st.sidebar.subheader("GPS (Optional)")
latitude = st.sidebar.number_input("Latitude", 0.0, format="%.6f")
longitude = st.sidebar.number_input("Longitude", 0.0, format="%.6f")

# =====================================================
# UPLOAD IMAGE MODE (DUAL OUTPUT)
# =====================================================
if mode == "Upload Image":
    uploaded = st.file_uploader("Upload Road Image", type=["jpg","png","jpeg"])

    if uploaded and st.button("Run Detection"):
        frame = np.array(Image.open(uploaded).convert("RGB"))

        # YOLO + Lane
        yolo_frame = frame.copy()
        boxes, scores, classes = run_tflite_inference(yolo_frame)
        draw_boxes(yolo_frame, boxes, scores, classes, latitude, longitude, conf)
        lanes = enhanced_lane_detection(yolo_frame)
        left_img = cv2.addWeighted(yolo_frame, 0.7, lanes, 0.5, 0)

        # Pothole Mask
        mask = pothole_mask_no_training(frame)
        mask_col = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Lane + YOLO Detection")
            st.image(left_img, channels="BGR")

        with col2:
            st.subheader("Pothole Mask (No Training)")
            st.image(mask_col, channels="BGR")

# =====================================================
# LIVE WEBCAM / DROIDCAM
# =====================================================
if mode == "Live Webcam / DroidCam":
    cam_url = st.text_input("Camera Source", "http://192.168.1.3:4747/video")
    run = st.checkbox("Start Camera")

    if run:
        cap = cv2.VideoCapture(cam_url)
        frame_box = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not accessible")
                break

            boxes, scores, classes = run_tflite_inference(frame)
            draw_boxes(frame, boxes, scores, classes, latitude, longitude, conf)
            lanes = enhanced_lane_detection(frame)
            blended = cv2.addWeighted(frame, 0.7, lanes, 0.5, 0)

            frame_box.image(blended, channels="BGR")

        cap.release()

# =====================================================
# DATABASE VIEW
# =====================================================
st.subheader("Detection Log")
df = pd.read_sql("SELECT * FROM pothole_events ORDER BY id DESC", conn)
st.dataframe(df)
