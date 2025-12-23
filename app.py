# =====================================================
# RQI – Road Quality Index
# YOLOv8 (TFLite) + Classical Lane Analysis + GIS Export
# =====================================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sqlite3
import datetime
import os
import pandas as pd
import json

st.set_page_config(page_title="RQI — YOLO + Lane + Map", layout="wide")

# =====================================================
# DATABASE (SQLite – lightweight & portable)
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
# STORAGE
# =====================================================
SNAPSHOT_FOLDER = "snapshots"
CLIP_FOLDER = "clips"
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)
os.makedirs(CLIP_FOLDER, exist_ok=True)

def insert_detection(class_name, confidence, snapshot_path, lat, lon):
    """
    Stores detection metadata for GIS / analytics usage
    """
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
# LOAD YOLO TFLITE MODEL
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
# NON-MAX SUPPRESSION (Pure NumPy)
# Removes duplicate overlapping detections
# =====================================================
def nms_numpy(boxes, scores, iou_thresh=0.45):
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes.T
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

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[np.where(iou <= iou_thresh)[0] + 1]

    return keep

# =====================================================
# YOLO TFLITE INFERENCE (Correct decoding)
# =====================================================
def run_tflite_inference(frame, conf_thres=0.25):
    """
    YOLOv8 TFLite decoding:
    - Letterbox resize
    - Center → corner box conversion
    - NMS filtering
    """

    ih, iw = frame.shape[:2]
    input_h, input_w = input_details[0]["shape"][1:3]

    # Letterbox resize (keeps aspect ratio)
    scale = min(input_w / iw, input_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized

    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    boxes, scores, classes = [], [], []

    for det in preds:
        if det[4] < conf_thres:
            continue

        xc, yc, w, h = det[:4]

        # Center → corner
        x1 = (xc - w / 2) * input_w
        y1 = (yc - h / 2) * input_h
        x2 = (xc + w / 2) * input_w
        y2 = (yc + h / 2) * input_h

        # Undo letterbox
        x1, x2 = x1 / scale, x2 / scale
        y1, y2 = y1 / scale, y2 / scale

        boxes.append([x1 / iw, y1 / ih, x2 / iw, y2 / ih])
        scores.append(float(det[4]))
        classes.append(int(det[5]))

    if not boxes:
        return [], [], []

    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)

    keep = nms_numpy(boxes, scores)
    return boxes[keep], scores[keep], classes[keep]

# =====================================================
# CLASSICAL CV – Missing Lane Detection (No ML)
# =====================================================
def detect_missing_lane(frame):
    """
    Detects missing lanes using edge continuity.
    Low continuity = missing lane
    """
    h, w = frame.shape[:2]
    roi = frame[int(h * 0.55):h, int(w * 0.15):int(w * 0.85)]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80,
        minLineLength=40, maxLineGap=60
    )

    total_len = 0
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            total_len += np.hypot(x2 - x1, y2 - y1)

    lane_score = total_len / (roi.shape[0] * 2)
    return lane_score < 0.35

# =====================================================
# DRAW DETECTIONS
# =====================================================
COLORS = [(0,255,0),(255,255,0),(255,0,255)]

def draw_boxes(frame, boxes, scores, classes, lat, lon, threshold):
    h, w = frame.shape[:2]
    coords = []

    for i, s in enumerate(scores):
        if s < threshold:
            continue

        x1, y1, x2, y2 = boxes[i]
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
        cid = int(classes[i])

        cv2.rectangle(frame, (x1,y1), (x2,y2), COLORS[cid % 3], 2)
        label = f"{CLASS_NAMES[cid]} {s:.2f}"
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cid % 3], 2)

        crop = frame[y1:y2, x1:x2]
        if crop.size:
            path = f"{SNAPSHOT_FOLDER}/{CLASS_NAMES[cid]}_{datetime.datetime.now().strftime('%H%M%S_%f')}.jpg"
            cv2.imwrite(path, crop)
            insert_detection(CLASS_NAMES[cid], s, path, lat, lon)
            coords.append((lat, lon))

    return coords

# =====================================================
# GEOJSON EXPORT
# =====================================================
def export_geojson(db_conn):
    df = pd.read_sql(
        "SELECT * FROM pothole_events WHERE latitude IS NOT NULL",
        db_conn
    )

    features = []
    for _, r in df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [r.longitude, r.latitude]
            },
            "properties": {
                "class": r.class_name,
                "confidence": r.confidence,
                "timestamp": r.timestamp,
                "snapshot": r.snapshot_path
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features
    }

# =====================================================
# UI
# =====================================================
st.title("🚧 Road Quality Index (RQI)")

mode = st.selectbox("Mode", ["Upload Image", "Live Camera"])
conf = st.sidebar.slider("Confidence", 0.05, 0.8, 0.25, 0.05)

lat = st.sidebar.number_input("Latitude", value=0.0)
lon = st.sidebar.number_input("Longitude", value=0.0)

# ================= IMAGE MODE =================
if mode == "Upload Image":
    img = st.file_uploader("Upload road image", ["jpg","png"])

    if img and st.button("Run Detection"):
        frame = cv2.cvtColor(np.array(Image.open(img)), cv2.COLOR_RGB2BGR)

        boxes, scores, classes = run_tflite_inference(frame, conf)
        coords = draw_boxes(frame, boxes, scores, classes, lat, lon, conf)

        if detect_missing_lane(frame):
            cv2.putText(frame, "MISSING LANE",
                        (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,0,255), 3)

        st.image(frame, channels="BGR")

# ================= EXPORT =================
st.subheader("🌍 Export")
if st.button("Export GeoJSON"):
    geo = export_geojson(conn)
    st.download_button(
        "Download GeoJSON",
        json.dumps(geo, indent=2),
        "rqi_detections.geojson",
        "application/geo+json"
    )

# ================= DATABASE VIEW =================
st.subheader("📋 Detection Log")
st.dataframe(pd.read_sql(
    "SELECT * FROM pothole_events ORDER BY id DESC", conn
))
