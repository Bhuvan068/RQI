import streamlit as st
import cv2
import numpy as np
from PIL import Image
import random
import sqlite3
import datetime
import os
import pandas as pd

st.set_page_config(page_title="RQI — YOLO + Lane + Red Potholes", layout="wide")

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
# LOAD TFLITE
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

# =====================================================
# LANE DETECTION
# =====================================================
def enhanced_lane_detection(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    yellow = cv2.inRange(hsv, (15, 70, 70), (35, 255, 255))
    mask = cv2.bitwise_or(white, yellow)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    return cv2.applyColorMap(mask, cv2.COLORMAP_HOT)

# =====================================================
# POTHOLE MASK (NO TRAINING, CLEAN)
# =====================================================
def pothole_mask_no_training(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    edges = cv2.Canny(blur, 60, 150)
    combined = cv2.bitwise_and(thresh, edges)

    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    final_mask = np.zeros_like(combined)
    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 400 or area > 20000:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if 0.2 < circularity < 0.85:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    return final_mask

# =====================================================
# COLOR POTHOLES RED
# =====================================================
def color_potholes_red(original, mask):
    overlay = original.copy()
    overlay[mask == 255] = [255, 0, 0]  # RED (BGR)
    return cv2.addWeighted(original, 0.6, overlay, 0.4, 0)

# =====================================================
# YOLO HELPERS
# =====================================================
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

COLORS = [(0,255,0),(255,255,0),(255,0,255)]

def draw_boxes(frame, boxes, scores, classes, lat, lon, threshold):
    h, w, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] < threshold:
            continue
        x1,y1,x2,y2 = boxes[i]
        x1,x2 = int(x1*w), int(x2*w)
        y1,y2 = int(y1*h), int(y2*h)

        cid = int(classes[i])
        cv2.rectangle(frame, (x1,y1), (x2,y2), COLORS[cid%3], 2)

        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            path = f"{SNAPSHOT_FOLDER}/{CLASS_NAMES[cid]}_{datetime.datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(path, crop)
            insert_detection(CLASS_NAMES[cid], float(scores[i]), path, lat, lon)

# =====================================================
# UI
# =====================================================
st.title("🚧 RQI — YOLO + Lane + Red Pothole Detection")

mode = st.selectbox("Select Mode", ["Upload Image", "Live Webcam / DroidCam"])
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

st.sidebar.subheader("GPS (optional)")
latitude = st.sidebar.number_input("Latitude", 0.0, format="%.6f")
longitude = st.sidebar.number_input("Longitude", 0.0, format="%.6f")

# =====================================================
# UPLOAD IMAGE MODE
# =====================================================
if mode == "Upload Image":
    uploaded = st.file_uploader("Upload Road Image", ["jpg","png","jpeg"])

    if uploaded and st.button("Run Detection"):
        frame = np.array(Image.open(uploaded).convert("RGB"))

        # LEFT: YOLO + LANE
        yolo_frame = frame.copy()
        boxes, scores, classes = run_tflite_inference(yolo_frame)
        draw_boxes(yolo_frame, boxes, scores, classes, latitude, longitude, conf)
        lanes = enhanced_lane_detection(yolo_frame)
        left_img = cv2.addWeighted(yolo_frame, 0.7, lanes, 0.5, 0)

        # RIGHT: RED POTHOLES ONLY
        pothole_mask = pothole_mask_no_training(frame)
        right_img = color_potholes_red(frame, pothole_mask)

        col1, col2 = st.columns(2)
        col1.image(left_img, caption="YOLO + Lane Overlay", channels="BGR")
        col2.image(right_img, caption="Potholes Highlighted (Red)", channels="BGR")

# =====================================================
# LIVE CAMERA MODE
# =====================================================
if mode == "Live Webcam / DroidCam":
    cam_url = st.text_input("Camera URL", "http://192.168.1.3:4747/video")
    start = st.checkbox("Start Camera")

    if start:
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
st.subheader("📋 Detection Log")
df = pd.read_sql("SELECT * FROM pothole_events ORDER BY id DESC", conn)
st.dataframe(df)
