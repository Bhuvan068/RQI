import streamlit as st
import cv2
import numpy as np
from PIL import Image
import random
import sqlite3
import datetime
import os
import pandas as pd
import streamlit.components.v1 as components

st.set_page_config(page_title="RQI — YOLO + Lane + Pothole + Heatmap", layout="wide")

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
# TFLITE
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
    white = cv2.inRange(hsv, (0,0,180), (180,40,255))
    yellow = cv2.inRange(hsv, (15,70,70), (35,255,255))
    mask = cv2.bitwise_or(white, yellow)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    return cv2.applyColorMap(mask, cv2.COLORMAP_HOT)

# =====================================================
# POTHOLE MASK (NO TRAINING)
# =====================================================
def pothole_mask_no_training(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, mask = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    return mask

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

COLORS = [(255,0,0),(0,255,0),(0,0,255)]

def draw_boxes(frame, boxes, scores, classes, lat, lon, threshold):
    h, w, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] < threshold:
            continue
        x1,y1,x2,y2 = boxes[i]
        x1,x2 = int(x1*w), int(x2*w)
        y1,y2 = int(y1*h), int(y2*h)

        cid = int(classes[i])
        color = COLORS[cid % len(COLORS)]
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            path = f"{SNAPSHOT_FOLDER}/{CLASS_NAMES[cid]}_{datetime.datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(path, crop)
            insert_detection(CLASS_NAMES[cid], float(scores[i]), path, lat, lon)

# =====================================================
# UI
# =====================================================
st.title("🚧 RQI — Ather-style Pothole Intelligence")

mode = st.selectbox("Mode", ["Upload Image", "Live Webcam / DroidCam"])
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

st.sidebar.subheader("GPS")
latitude = st.sidebar.number_input("Latitude", 0.0, format="%.6f")
longitude = st.sidebar.number_input("Longitude", 0.0, format="%.6f")

# =====================================================
# UPLOAD IMAGE MODE
# =====================================================
if mode == "Upload Image":
    file = st.file_uploader("Upload Road Image", ["jpg","png","jpeg"])
    if file and st.button("Run Detection"):
        frame = np.array(Image.open(file).convert("RGB"))

        yolo_frame = frame.copy()
        boxes,scores,classes = run_tflite_inference(yolo_frame)
        draw_boxes(yolo_frame, boxes, scores, classes, latitude, longitude, conf)

        lanes = enhanced_lane_detection(yolo_frame)
        left = cv2.addWeighted(yolo_frame,0.7,lanes,0.5,0)

        mask = pothole_mask_no_training(frame)
        right = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        c1,c2 = st.columns(2)
        c1.image(left, caption="YOLO + Lane", channels="BGR")
        c2.image(right, caption="Pothole Mask", channels="BGR")

# =====================================================
# LIVE CAMERA
# =====================================================
if mode == "Live Webcam / DroidCam":
    url = st.text_input("Camera URL", "http://192.168.1.3:4747/video")
    start = st.checkbox("Start Camera")
    if start:
        cap = cv2.VideoCapture(url)
        box = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            boxes,scores,classes = run_tflite_inference(frame)
            draw_boxes(frame, boxes, scores, classes, latitude, longitude, conf)
            lanes = enhanced_lane_detection(frame)
            blend = cv2.addWeighted(frame,0.7,lanes,0.5,0)
            box.image(blend, channels="BGR")
        cap.release()

# =====================================================
# DATABASE TABLE
# =====================================================
st.subheader("📋 Detection Log")
df = pd.read_sql("SELECT * FROM pothole_events ORDER BY id DESC", conn)
st.dataframe(df)

# =====================================================
# OPENSTREETMAP + HEATMAP
# =====================================================
st.subheader("🗺️ Pothole Density Map (Heatmap)")

map_df = pd.read_sql("""
SELECT latitude, longitude, class_name, timestamp
FROM pothole_events
WHERE latitude != 0 AND longitude != 0
""", conn)

if not map_df.empty:
    heat_points = ",".join([
        f"[{r.latitude}, {r.longitude}, 0.8]"
        for _, r in map_df.iterrows()
    ])

    components.html(f"""
    <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet.heat/dist/leaflet-heat.js"></script>

    <div id="map" style="height:520px;"></div>

    <script>
      var map = L.map('map').setView(
        [{map_df.latitude.mean()}, {map_df.longitude.mean()}], 15
      );

      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        attribution: '© OpenStreetMap'
      }}).addTo(map);

      var heat = L.heatLayer([{heat_points}], {{
        radius: 25,
        blur: 15,
        maxZoom: 17
      }}).addTo(map);
    </script>
    """, height=540)
else:
    st.info("No GPS data available for map.")
