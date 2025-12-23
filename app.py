import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sqlite3
import datetime
import os
import pandas as pd

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

COLORS = [(0,255,0),(255,255,0),(255,0,255)]

# =====================================================
# LANE DETECTION (color thresholding)
# =====================================================
def enhanced_lane_detection(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    yellow = cv2.inRange(hsv, (15, 70, 70), (35, 255, 255))
    mask = cv2.bitwise_or(white, yellow)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return mask

# =====================================================
# YOLO + TFLite (with NMS)
# =====================================================
def preprocess_for_tflite(frame):
    h, w = input_details[0]["shape"][1:3]
    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def run_tflite_inference(frame, conf_thres=0.25, iou_thres=0.45):
    interpreter.set_tensor(input_details[0]["index"], preprocess_for_tflite(frame))
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    boxes, scores, classes = [], [], []

    for det in preds:
        conf = det[4]
        if conf < conf_thres:
            continue
        xc, yc, w, h = det[:4]
        x1 = xc - w/2
        y1 = yc - h/2
        x2 = xc + w/2
        y2 = yc + h/2
        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        classes.append(int(det[5]))

    # Convert to numpy
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)

    # --- Apply NMS ---
    if len(boxes) > 0:
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]

        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]

    # Scale to 0-1 relative to frame
    h_frame, w_frame = frame.shape[:2]
    boxes[:,0] /= w_frame
    boxes[:,2] /= w_frame
    boxes[:,1] /= h_frame
    boxes[:,3] /= h_frame

    return boxes, scores, classes

# =====================================================
# DRAW BOXES (fixed scaling)
# =====================================================
def draw_boxes(frame, boxes, scores, classes, lat, lon, threshold):
    coords = []
    for i, s in enumerate(scores):
        if s < threshold:
            continue
        x1, y1, x2, y2 = boxes[i]
        x1 = int(x1 * frame.shape[1])
        x2 = int(x2 * frame.shape[1])
        y1 = int(y1 * frame.shape[0])
        y2 = int(y2 * frame.shape[0])
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
# GET GPS
# =====================================================
st.sidebar.subheader("📍 GPS Coordinates (Required for Map)")
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
st.title("🚧 RQI (Road Quality Index)")
mode = st.selectbox("Select Mode", ["Upload Image", "Live Webcam / DroidCam"])
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

if mode == "Upload Image":
    img = st.file_uploader("Upload road image", ["jpg","jpeg","png"])
    if img and st.button("Run Detection"):
        frame = cv2.cvtColor(np.array(Image.open(img)), cv2.COLOR_RGB2BGR)
        overlay = frame.copy()

        # --- Inference ---
        boxes, scores, classes = run_tflite_inference(overlay, conf_thres=conf)
        lane_mask = enhanced_lane_detection(overlay)

        lat, lon = get_gps()
        coords = []
        if lat is not None:
            coords = draw_boxes(overlay, boxes, scores, classes, lat, lon, conf)

        # --- Lane continuity analysis ---
        lane_score = np.sum(lane_mask) / 255  # simple proxy for continuity
        missing_lane = lane_score < 100  # threshold, tune as needed

        # --- Layout columns ---
        col_img, col_stats = st.columns([3,1])
        with col_img:
            final_img = cv2.addWeighted(
                overlay, 0.7,
                cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR), 0.5, 0
            )
            st.image(final_img, caption="YOLO + Lane Overlay", channels="BGR")

        with col_stats:
            st.subheader("📊 Analysis")
            st.write(f"Detections: {len(boxes)}")
            st.write(f"Lane continuity score: {lane_score:.2f}")
            if missing_lane:
                st.error("❌ Missing Lane")
            else:
                st.success("✅ Lane OK")

        # --- Map ---
        if coords:
            st.subheader("🗺️ Pothole Location")
            st.map(pd.DataFrame(coords, columns=["lat","lon"]))

# =====================================================
# DATABASE VIEW
# =====================================================
st.subheader("📋 Detection Log")
st.dataframe(pd.read_sql(
    "SELECT * FROM pothole_events ORDER BY id DESC", conn
))


