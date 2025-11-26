import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# Try TFLite interpreter
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


# ---------------------------
# Load TFLite Model
# ---------------------------
MODEL_PATH = "best_float16.tflite"

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img_h = input_details[0]['shape'][1]
img_w = input_details[0]['shape'][2]


# ---------------------------
# Detection Function
# ---------------------------
def detect(frame):
    img = cv2.resize(frame, (img_w, img_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    results = interpreter.get_tensor(output_details[0]['index'])
    return results[0]


# ---------------------------
# Highlight Lane
# ---------------------------
def highlight_lane(frame, x1, y1, x2, y2):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
    return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="IP Webcam Lane Detection", layout="wide")

st.title("📡 Real-Time Lane Detection (YOLO + TFLite + IP Webcam)")
st.markdown("Supports **Laptop webcam** & **IP Webcam (Android IP Webcam app)**")

st.sidebar.header("Camera Settings")

camera_type = st.sidebar.selectbox(
    "Select Camera Source",
    ["Laptop Webcam", "IP Webcam"]
)

ip_url = ""
if camera_type == "IP Webcam":
    ip_url = st.sidebar.text_input(
        "Enter IP Webcam URL",
        "http://192.168.1.5:8080/video"
    )

run_btn = st.sidebar.button("▶ Start Detection")


# ---------------------------
# Start Video Stream
# ---------------------------
frame_placeholder = st.empty()

if run_btn:

    if camera_type == "Laptop Webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(ip_url)

    if not cap.isOpened():
        st.error(" Unable to open camera stream.")
        st.stop()

    st.sidebar.success(" Running... Press STOP to end.")

    stop_btn = st.sidebar.button(" Stop")

    while True:

        if stop_btn:
            break

        ret, frame = cap.read()
        if not ret:
            st.error("Frame not received — check camera!")
            break

        detections = detect(frame)

        h, w, _ = frame.shape

        for det in detections:
            x1, y1, x2, y2, score, cls = det

            if score < 0.5:
                continue

            # Convert normalized → pixel
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)

            # Lane class = 0 (change if needed)
            if int(cls) == 0:
                frame = highlight_lane(frame, x1, y1, x2, y2)
                cv2.putText(frame, f"Lane {score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Obj {score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Convert BGR → RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame, channels="RGB")

    cap.release()
