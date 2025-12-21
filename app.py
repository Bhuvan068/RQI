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

# Install streamlit_javascript if not already installed
# pip install streamlit-javascript
from streamlit_javascript import st_javascript

st.set_page_config(page_title="RQI — YOLO + DeepLaneNet + Map", layout="wide")

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
# LOAD TFLITE YOLO MODEL
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
# LANE DETECTION FUNCTION (placeholder)
# =====================================================
def deep_lane_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)
    lane_img = np.zeros_like(frame)
    
    if lines is None or len(lines)==0:
        return lane_img, False
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(lane_img,(x1,y1),(x2,y2),(0,255,0),5)
    return lane_img, True

# =====================================================
# YOLO HELPERS
# =====================================================
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

COLORS = [(0,255,0),(255,255,0),(255,0,255)]

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

# =====================================================
# GET USER LOCATION VIA BROWSER
# =====================================================
location_js = """
async function getLocation(){
  return new Promise((resolve,reject)=>{
    if(navigator.geolocation){
        navigator.geolocation.getCurrentPosition(
            pos => resolve([pos.coords.latitude,pos.coords.longitude]),
            err => resolve([20.5937, 78.9629]) // fallback to India
        );
    } else { resolve([20.5937, 78.9629]); }
  });
}
getLocation();
"""
user_lat, user_lon = st_javascript(location_js)
st.sidebar.write(f"Detected Location: Latitude={user_lat:.6f}, Longitude={user_lon:.6f}")

# =====================================================
# UI
# =====================================================
st.title("🚧 RQI — YOLO + DeepLaneNet + Map")

mode = st.selectbox("Select Mode", ["Upload Image", "Live Webcam / DroidCam"])
conf = st.sidebar.slider("Confidence Threshold",0.1,1.0,0.3,0.05)

# ================= UPLOAD IMAGE MODE =================
if mode=="Upload Image":
    uploaded = st.file_uploader("Upload Road Image",["jpg","png","jpeg"])
    if uploaded and st.button("Run Detection"):
        frame = np.array(Image.open(uploaded).convert("RGB"))
        
        lane_img, lane_found = deep_lane_detection(frame)
        if not lane_found:
            st.warning("⚠️ Lane Missing!")
            insert_detection("Lane Missing", 0.0, "", user_lat, user_lon)
        
        yolo_frame = frame.copy()
        boxes,scores,classes = run_tflite_inference(yolo_frame)
        draw_boxes(yolo_frame, boxes, scores, classes, user_lat, user_lon, conf)

        col1, col2 = st.columns(2)
        col1.image(lane_img, caption="DeepLaneNet Output", channels="BGR")
        col2.image(yolo_frame, caption="YOLO Detection", channels="BGR")

# ================= LIVE CAMERA MODE =================
if mode=="Live Webcam / DroidCam":
    st.info("Use a ngrok public URL here if running on Streamlit Cloud")
    cam_url = st.text_input("Camera URL","http://192.168.1.3:4747/video")
    start = st.checkbox("Start Camera")

    if start:
        col1, col2 = st.columns(2)
        cap = cv2.VideoCapture(cam_url)
        if not cap.isOpened():
            st.error("Cannot access camera. Check URL or ngrok link.")
        else:
            while cap.isOpened():
                ret,frame = cap.read()
                if not ret:
                    st.error("Camera not accessible")
                    break

                lane_img, lane_found = deep_lane_detection(frame)
                if not lane_found:
                    insert_detection("Lane Missing", 0.0, "", user_lat, user_lon)

                yolo_frame = frame.copy()
                boxes,scores,classes = run_tflite_inference(yolo_frame)
                draw_boxes(yolo_frame, boxes, scores, classes, user_lat, user_lon, conf)

                col1.image(lane_img, caption="DeepLaneNet Output", channels="BGR")
                col2.image(yolo_frame, caption="YOLO Detection", channels="BGR")

            cap.release()

# ================= DATABASE VIEW =================
st.subheader("📋 Detection Log")
df = pd.read_sql("SELECT * FROM pothole_events ORDER BY id DESC", conn)
st.dataframe(df)

# ================= MAP VIEW =================
st.subheader("🗺️ Map of Detections")
map_center = [user_lat, user_lon]
m = folium.Map(location=map_center, zoom_start=14)

for _, row in df.iterrows():
    if row['class_name']=="Lane Missing":
        folium.Marker([row['latitude'], row['longitude']],
                      popup="Lane Missing",
                      icon=folium.Icon(color='orange', icon='exclamation-sign')).add_to(m)
    else:
        folium.Marker([row['latitude'], row['longitude']],
                      popup=f"{row['class_name']} ({row['confidence']:.2f})",
                      icon=folium.Icon(color='red', icon='info-sign')).add_to(m)

st_folium(m, width=700, height=500)
