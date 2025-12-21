import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sqlite3
import datetime
import os
import pandas as pd
from streamlit_folium import st_folium
import folium
import tensorflow as tf

st.set_page_config(page_title="RQI — YOLO + Lane Detection", layout="wide")

# ================= DATABASE =================
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

# ================= SNAPSHOTS =================
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

# ================= UPLOAD TFLITE MODEL =================
st.sidebar.subheader("Upload YOLO TFLite Model")
model_file = st.sidebar.file_uploader("Choose .tflite file", type=["tflite"])

if model_file:
    with open("best_float16.tflite", "wb") as f:
        f.write(model_file.read())
    MODEL_PATH = "best_float16.tflite"
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
else:
    st.warning("Upload YOLO .tflite model to continue")
    st.stop()

CLASS_NAMES = ["Pothole", "Crack", "Faded Lane"]
COLORS = [(0,255,0),(255,255,0),(255,0,255)]

# ================= YOLO HELPERS =================
def preprocess_for_tflite(frame):
    h, w = input_details[0]["shape"][1:3]
    img = cv2.resize(frame,(w,h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.0
    return np.expand_dims(img, axis=0)

def run_tflite_inference(frame):
    inp = preprocess_for_tflite(frame)
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])[0]
    if out.size==0:
        return [],[],[]
    return out[:,:4], out[:,4], out[:,5]

def draw_boxes(frame, boxes, scores, classes, lat, lon, threshold):
    h,w,_ = frame.shape
    for i in range(len(scores)):
        if scores[i]<threshold:
            continue
        x1,y1,x2,y2 = boxes[i]
        x1,x2 = int(x1*w), int(x2*w)
        y1,y2 = int(y1*h), int(y2*h)
        cid = int(classes[i])
        cv2.rectangle(frame,(x1,y1),(x2,y2),COLORS[cid%3],2)
        crop = frame[y1:y2,x1:x2]
        if crop.size>0:
            path = f"{SNAPSHOT_FOLDER}/{CLASS_NAMES[cid]}_{datetime.datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(path, crop)
            insert_detection(CLASS_NAMES[cid], float(scores[i]), path, lat, lon)

# ================= LANE DETECTION =================
def enhanced_lane_detection(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0,0,180),(180,60,255))   # medium threshold
    yellow = cv2.inRange(hsv, (15,100,100),(35,255,255))
    mask = cv2.bitwise_or(white,yellow)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    lane_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return lane_img

# ================= UI =================
st.title("🚧 RQI — YOLO + Lane Detection")

mode = st.selectbox("Select Mode", ["Upload Image", "Live Webcam / DroidCam"])
conf = st.sidebar.slider("Confidence Threshold",0.1,1.0,0.3,0.05)

st.sidebar.subheader("GPS (optional)")
latitude = st.sidebar.number_input("Latitude",0.0,format="%.6f")
longitude = st.sidebar.number_input("Longitude",0.0,format="%.6f")

# ================= UPLOAD IMAGE MODE =================
if mode=="Upload Image":
    uploaded = st.file_uploader("Upload Road Image",["jpg","png","jpeg"])
    if uploaded and st.button("Run Detection"):
        frame = np.array(Image.open(uploaded).convert("RGB"))

        # YOLO
        yolo_frame = frame.copy()
        boxes,scores,classes = run_tflite_inference(yolo_frame)
        draw_boxes(yolo_frame, boxes, scores, classes, latitude, longitude, conf)

        # Lane Detection
        lane_img = enhanced_lane_detection(frame)

        col1,col2 = st.columns(2)
        col1.image(lane_img, caption="Lane Detection (B/W)", channels="BGR")
        col2.image(yolo_frame, caption="YOLO Detection", channels="BGR")

# ================= LIVE CAMERA MODE =================
if mode=="Live Webcam / DroidCam":
    st.info("Use a ngrok public URL here if running on Streamlit Cloud")
    cam_url = st.text_input("Camera URL","http://192.168.1.3:4747/video")
    start = st.checkbox("Start Camera")

    if start:
        exit_button = st.button("Stop Detection")
        frame_box1 = st.empty()
        frame_box2 = st.empty()
        cap = cv2.VideoCapture(cam_url)
        if not cap.isOpened():
            st.error("Cannot access camera. Check URL or ngrok link.")
        else:
            while cap.isOpened():
                ret,frame = cap.read()
                if not ret:
                    st.error("Camera not accessible")
                    break

                # YOLO
                yolo_frame = frame.copy()
                boxes,scores,classes = run_tflite_inference(yolo_frame)
                draw_boxes(yolo_frame, boxes, scores, classes, latitude, longitude, conf)

                # Lane
                lane_img = enhanced_lane_detection(frame)

                frame_box1.image(lane_img, caption="Lane Detection (B/W)", channels="BGR")
                frame_box2.image(yolo_frame, caption="YOLO Detection", channels="BGR")

                if exit_button:
                    break
            cap.release()

# ================= DATABASE VIEW =================
st.subheader("📋 Detection Log")
df = pd.read_sql("SELECT * FROM pothole_events ORDER BY id DESC", conn)
st.dataframe(df)

# ================= MAP VIEW =================
st.subheader("🗺️ Map of Detections")
map_center = [latitude, longitude] if latitude!=0 and longitude!=0 else [20.5937,78.9629]
m = folium.Map(location=map_center, zoom_start=14)

for _, row in df.iterrows():
    folium.Marker([row['latitude'], row['longitude']],
                  popup=f"{row['class_name']} ({row['confidence']:.2f})",
                  icon=folium.Icon(color='red', icon='info-sign')).add_to(m)

st_folium(m, width=700, height=500)
