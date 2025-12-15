import streamlit as st
import cv2
import numpy as np
from PIL import Image
import random
import sqlite3
import datetime

st.set_page_config(page_title="RQI — Real-Time YOLO + Lane Detection", layout="wide")

# --------------------------
# SQLite Database Setup
# --------------------------
conn = sqlite3.connect("potholes.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS pothole_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    class_name TEXT,
    confidence REAL,
    latitude REAL,
    longitude REAL
)
""")
conn.commit()

def insert_detection(class_name, confidence, latitude=None, longitude=None):
    timestamp = datetime.datetime.now().isoformat()
    cursor.execute("""
    INSERT INTO pothole_events (timestamp, class_name, confidence, latitude, longitude)
    VALUES (?, ?, ?, ?, ?)
    """, (timestamp, class_name, confidence, latitude, longitude))
    conn.commit()

# --------------------------
# Try import tflite runtime first, fallback to tensorflow
# --------------------------
Interpreter = None
tflite_import_error = None

try:
    from tflite_runtime.interpreter import Interpreter
    st.sidebar.info("Using tflite_runtime.Interpreter")
except Exception as e1:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        st.sidebar.info("Using tensorflow.lite.Interpreter")
    except Exception as e2:
        Interpreter = None
        tflite_import_error = (e1, e2)

# --------------------------
# CONFIG
# --------------------------
MODEL_PATH = "best_float16.tflite"
CONF_THRESHOLD = 0.5
CLASS_NAMES = ["Pothole", "Crack", "Faded Lane"]

# --------------------------
# MODEL LOADING
# --------------------------
interpreter = None
input_details = None
output_details = None
model_loaded = False

if Interpreter is not None:
    try:
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        model_loaded = True
    except Exception as ex:
        st.error(f"Model failed to load: {ex}")

# --------------------------
# LANE DETECTION (ULTRA BRIGHT)
# --------------------------
def enhanced_lane_detection(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    white_lower = np.array([0, 0, 180])
    white_upper = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)

    yellow_lower = np.array([15, 70, 70])
    yellow_upper = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
    kernel = np.ones((5, 5), np.uint8)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)

    lane_color = cv2.applyColorMap(lane_mask, cv2.COLORMAP_HOT)
    return lane_color

# --------------------------
# DETECTION UTILITIES
# --------------------------
def generate_colors(num_classes):
    random.seed(42)
    return [tuple([random.randint(0,255) for _ in range(3)]) for __ in range(num_classes)]

COLORS = generate_colors(len(CLASS_NAMES))

def preprocess_for_tflite(frame):
    h, w = input_details[0]["shape"][1:3]
    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def draw_boxes(frame, boxes, scores, classes, threshold=0.5):
    h, w, _ = frame.shape
    for i in range(len(scores)):
        if float(scores[i]) < threshold:
            continue

        xmin = int(boxes[i][0] * w)
        ymin = int(boxes[i][1] * h)
        xmax = int(boxes[i][2] * w)
        ymax = int(boxes[i][3] * h)

        class_id = int(classes[i])
        class_id = class_id if class_id < len(CLASS_NAMES) else 0
        color = COLORS[class_id]
        label = f"{CLASS_NAMES[class_id]}: {scores[i]:.2f}"

        # Draw on frame
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, label, (xmin, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Insert into SQL
        insert_detection(CLASS_NAMES[class_id], float(scores[i]))

def run_tflite_inference(frame):
    inp = preprocess_for_tflite(frame)
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()

    out = interpreter.get_tensor(output_details[0]["index"])
    out = np.array(out)

    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]

    if out.size == 0:
        return np.zeros((0,4)), np.zeros((0,)), np.zeros((0,))

    boxes = out[:, :4]
    scores = out[:, 4]
    classes = out[:, 5]
    return boxes, scores, classes

# --------------------------
# STREAMLIT UI
# --------------------------
st.title("RQI — Real-Time YOLO + Lane Detection (TFLite)")

if not model_loaded:
    st.error("Model failed to load.")
    st.stop()

mode = st.selectbox("Choose Mode", ["Upload Image", "Live Webcam"])

# =====================================================
# MODE 1 — Upload Image
# =====================================================
if mode == "Upload Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded:
        frame = np.array(Image.open(uploaded).convert("RGB"))
        st.image(frame, caption="Original Image", channels="RGB")

        if st.button("Run Detection"):
            lanes = enhanced_lane_detection(frame)
            blended = cv2.addWeighted(frame, 0.7, lanes, 0.5, 0)
            boxes, scores, classes = run_tflite_inference(blended)
            draw_boxes(blended, boxes, scores, classes, threshold=CONF_THRESHOLD)
            st.image(blended, caption="YOLO + Lane Highlight", channels="BGR")

# =====================================================
# MODE 2 — Live Webcam
# =====================================================
if mode == "Live Webcam":
    if "run_cam" not in st.session_state:
        st.session_state.run_cam = False

    start = st.button("Start Webcam")
    stop = st.button("Stop Webcam")

    if start:
        st.session_state.run_cam = True
    if stop:
        st.session_state.run_cam = False

    cam = cv2.VideoCapture(0)  # Replace 0 with DroidCam IP if needed
    stframe = st.empty()

    while st.session_state.run_cam:
        ret, frame = cam.read()
        if not ret:
            st.error("Camera not found.")
            break

        lanes = enhanced_lane_detection(frame)
        blended = cv2.addWeighted(frame, 0.7, lanes, 0.5, 0)

        try:
            boxes, scores, classes = run_tflite_inference(blended)
            draw_boxes(blended, boxes, scores, classes, threshold=CONF_THRESHOLD)
        except:
            pass

        stframe.image(blended, channels="BGR")

    cam.release()

# =====================================================
# Display Detection Log from SQLite
# =====================================================
import pandas as pd
st.subheader("Detection Log")
df = pd.read_sql("SELECT * FROM pothole_events ORDER BY id DESC", conn)
st.dataframe(df)
