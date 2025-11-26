import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
import os

# ---------------------------
# Detect if running on Streamlit Cloud
# ---------------------------
ON_CLOUD = "STREAMLIT_RUNTIME" in os.environ

# ---------------------------
# Load TFLite Model
# ---------------------------
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

MODEL_PATH = "best_float16.tflite"

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
img_h = input_details[0]["shape"][1]
img_w = input_details[0]["shape"][2]


# ---------------------------
# YOLO Detection
# ---------------------------
def detect(frame):
    img = cv2.resize(frame, (img_w, img_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0).astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    return interpreter.get_tensor(output_details[0]["index"])[0]


# ---------------------------
# Lane Highlight Overlay
# ---------------------------
def highlight_lane(frame, x1, y1, x2, y2):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
    return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Hybrid Lane Detection", layout="wide")
st.title("🛣️ Hybrid Lane Detection (Local + Cloud)")

st.sidebar.header("Camera Settings")

mode = st.sidebar.selectbox(
    "Select Camera Source",
    ["Laptop Webcam", "IP Webcam", "Demo Video"]
)

ip_url = ""
if mode == "IP Webcam":
    ip_url = st.sidebar.text_input("Enter IP Webcam URL (shot.jpg):",
                                   "http://192.168.1.10:8080/shot.jpg")

start_btn = st.sidebar.button("▶ Start Detection")

frame_placeholder = st.empty()


# ---------------------------
# Hybrid Logic
# ---------------------------
if start_btn:

    # ----------------------------
    # If on cloud → force DEMO mode
    # ----------------------------
    if ON_CLOUD:
        st.warning("⚠ Running on Streamlit Cloud. Local webcams are not accessible.")
        st.info("Demo video is used instead.")

        mode = "Demo Video"

    # ----------------------------
    # Laptop Webcam (LOCAL ONLY)
    # ----------------------------
    if mode == "Laptop Webcam":
        if ON_CLOUD:
            st.error("Laptop webcam cannot be accessed from Cloud.")
            st.stop()

        cap = cv2.VideoCapture(0)

    # ----------------------------
    # IP Webcam (LOCAL ONLY)
    # ----------------------------
    elif mode == "IP Webcam":
        if ON_CLOUD:
            st.error("Local IP webcams cannot be accessed from Cloud.")
            st.stop()

        cap = ip_url  # use requests.get later

    # ----------------------------
    # Demo Video (works everywhere)
    # ----------------------------
    elif mode == "Demo Video":
        demo_path = "demo_lane.mp4"
        if not os.path.exists(demo_path):
            st.error("Demo video missing! Please upload demo_lane.mp4")
            st.stop()

        cap = cv2.VideoCapture(demo_path)

    stop_btn = st.sidebar.button("⛔ Stop")

    # ----------------------------
    # MAIN LOOP
    # ----------------------------
    while True:

        if stop_btn:
            break

        # IP camera (LOCAL)
        if mode == "IP Webcam":
            try:
                img_resp = requests.get(ip_url, timeout=2)
                img_arr = np.array(bytearray(img_resp.content), np.uint8)
                frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            except:
                st.error("Failed to connect to IP Webcam.")
                break

        # Laptop webcam / Demo Video
        else:
            ret, frame = cap.read()
            if not ret:
                st.error("Frame read failed.")
                break

        # Run YOLO Lite
        dets = detect(frame)

        h, w, _ = frame.shape

        # Draw detections
        for d in dets:
            x1, y1, x2, y2, score, cls = d

            if score < 0.4:
                continue

            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)

            if int(cls) == 0:  # lane class
                frame = highlight_lane(frame, x1, y1, x2, y2)
                cv2.putText(frame, "Lane", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Streamlit Display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels="RGB")

    if mode != "IP Webcam":
        cap.release()
