import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sqlite3
import datetime
import os
import pandas as pd
import time

st.set_page_config(page_title="RQI — YOLO + Lane + Map", layout="wide")

# =====================================================
# DATABASE
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
# SNAPSHOTS
# =====================================================
SNAPSHOT_FOLDER = "snapshots"
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

def insert_detection(class_name, confidence, snapshot_path, lat, lon):
    cursor.execute("""
        INSERT INTO pothole_events
        (timestamp, class_name, confidence, snapshot_path, latitude, longitude)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.datetime.now().isoformat(),
        class_name,
        confidence,
        snapshot_path,
        lat,
        lon
    ))
    conn.commit()

# =====================================================
# LOAD TFLITE MODEL
# =====================================================
try:
    from tflite_runtime.interpreter import Interpreter
except:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

MODEL_PATH = "best_float16.tflite"
CLASS_NAMES = ["Pothole", "Crack", "Faded Lane"]

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.success(f"Loaded model: {MODEL_PATH}")

# =====================================================
# MISSING LANE DETECTION (EDGE-BASED)
# =====================================================
def detect_missing_lane_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    h, w = edges.shape
    roi = np.zeros_like(edges)
    roi[int(h*0.5):h, :] = 255
    edges = cv2.bitwise_and(edges, roi)

    gap_mask = np.zeros_like(edges)
    missing = False

    strip_w = 5
    strip_h = 12
    edge_thresh = 5

    for x in range(60, w-60, strip_w):
        for y in range(int(h*0.55), h, strip_h):
            block = edges[y:y+strip_h, x:x+strip_w]
            if np.sum(block > 0) < edge_thresh:
                gap_mask[y:y+strip_h, x:x+strip_w] = 255
                missing = True

    gap_mask = cv2.dilate(gap_mask, np.ones((3,3),np.uint8), 1)

    overlay = frame.copy()
    overlay[gap_mask == 255] = [255, 0, 0]

    return overlay, missing

# =====================================================
# YOLO
# =====================================================
def preprocess_for_tflite(frame):
    h, w = input_details[0]["shape"][1:3]
    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.0
    return np.expand_dims(img, axis=0)

def run_tflite_inference(frame):
    interpreter.set_tensor(input_details[0]["index"], preprocess_for_tflite(frame))
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])[0]
    if out.size == 0:
        return [], [], []
    return out[:, :4], out[:, 4], out[:, 5]

COLORS = [(0,255,0),(255,255,0),(255,0,255)]

def draw_boxes(frame, boxes, scores, classes, lat, lon, threshold):
    h, w, _ = frame.shape
    coords = []

    for i, s in enumerate(scores):
        if s < threshold:
            continue
        x1,y1,x2,y2 = boxes[i]
        x1,x2 = int(x1*w), int(x2*w)
        y1,y2 = int(y1*h), int(y2*h)
        cid = int(classes[i])

        cv2.rectangle(frame,(x1,y1),(x2,y2),COLORS[cid%3],2)

        crop = frame[y1:y2, x1:x2]
        if crop.size:
            path = f"{SNAPSHOT_FOLDER}/{CLASS_NAMES[cid]}_{datetime.datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(path, crop)
            insert_detection(CLASS_NAMES[cid], float(s), path, lat, lon)
            coords.append((lat, lon))
    return coords

# =====================================================
# UI
# =====================================================
st.title("🚧 RQI (Road Quality Index)")
mode = st.selectbox("Select Mode", ["Upload Image", "Live Webcam / DroidCam"])
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

lat_txt = st.sidebar.text_input("Latitude")
lon_txt = st.sidebar.text_input("Longitude")

def get_gps():
    try:
        return float(lat_txt), float(lon_txt)
    except:
        return None, None

# =====================================================
# IMAGE MODE
# =====================================================
if mode == "Upload Image":
    img = st.file_uploader("Upload road image", ["jpg","jpeg","png"])
    if img and st.button("Run Detection"):
        frame = np.array(Image.open(img).convert("RGB"))
        overlay = frame.copy()

        boxes, scores, classes = run_tflite_inference(overlay)
        lat, lon = get_gps()
        if lat is not None:
            draw_boxes(overlay, boxes, scores, classes, lat, lon, conf)

        lane_overlay, missing = detect_missing_lane_edges(overlay)

        if missing and lat is not None:
            path = f"{SNAPSHOT_FOLDER}/FadedLane_{datetime.datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(path, lane_overlay)
            insert_detection("Faded Lane", 1.0, path, lat, lon)
            st.warning("⚠️ Missing lane detected")

        st.image(lane_overlay, channels="BGR")

# =====================================================
# LIVE CAMERA MODE
# =====================================================
if mode == "Live Webcam / DroidCam":
    cam_url = st.text_input("Camera Stream URL")
    start = st.checkbox("▶ Start Camera")
    pause = st.checkbox("⏸ Pause")

    frame_box = st.empty()
    last_save = 0

    if start and cam_url:
        cap = cv2.VideoCapture(cam_url)
        lat, lon = get_gps()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if not pause:
                boxes, scores, classes = run_tflite_inference(frame)
                if lat is not None:
                    draw_boxes(frame, boxes, scores, classes, lat, lon, conf)

                lane_overlay, missing = detect_missing_lane_edges(frame)

                if missing and lat is not None and time.time() - last_save > 5:
                    path = f"{SNAPSHOT_FOLDER}/FadedLane_{datetime.datetime.now().strftime('%H%M%S')}.jpg"
                    cv2.imwrite(path, lane_overlay)
                    insert_detection("Faded Lane", 1.0, path, lat, lon)
                    last_save = time.time()

                frame_box.image(lane_overlay, channels="BGR")

        cap.release()

# =====================================================
# DATABASE VIEW
# =====================================================
st.subheader("📋 Detection Log")
st.dataframe(pd.read_sql("SELECT * FROM pothole_events ORDER BY id DESC", conn))
