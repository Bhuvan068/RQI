import streamlit as st
import cv2
import numpy as np
from PIL import Image
import random
import traceback

st.set_page_config(page_title="TFLite Detection (with boxes & labels)", layout="wide")

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
load_error = None

if Interpreter is not None:
    try:
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        model_loaded = True
    except Exception as ex:
        load_error = ex
        interpreter = None

# --------------------------
# UTILITIES
# --------------------------
def generate_colors(num_classes):
    random.seed(42)
    return [tuple([int(random.randint(0,255)) for _ in range(3)]) for __ in range(num_classes)]

COLORS = generate_colors(len(CLASS_NAMES))

def preprocess_for_tflite(frame, desired_shape=None):
    if desired_shape is None:
        h, w = input_details[0]['shape'][1:3]
    else:
        h, w = desired_shape

    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def draw_boxes(frame, boxes, scores, classes, threshold=0.5):
    h, w, _ = frame.shape
    for i in range(len(scores)):
        score = float(scores[i])
        if score < threshold:
            continue

        box = boxes[i]
        xmin = int(box[0] * w)
        ymin = int(box[1] * h)
        xmax = int(box[2] * w)
        ymax = int(box[3] * h)

        class_id = int(classes[i])
        class_id = class_id if class_id < len(CLASS_NAMES) else 0
        color = COLORS[class_id]

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (xmin, ymin - label_h - baseline - 6),
                      (xmin + label_w + 6, ymin), color, -1)
        cv2.putText(frame, label, (xmin + 3, ymin - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def run_tflite_inference(frame):
    if not model_loaded:
        raise RuntimeError("Model not loaded.")

    inp = preprocess_for_tflite(frame)
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])

    out = np.array(out)
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]

    if out.size == 0:
        return np.zeros((0,4)), np.zeros((0,)), np.zeros((0,))

    if out.shape[1] >= 6:
        boxes = out[:, :4]
        scores = out[:, 4]
        classes = out[:, 5]
        return boxes, scores, classes

    raise RuntimeError(f"Unexpected model output shape: {out.shape}")

# --------------------------
# UI
# --------------------------
st.title("YOLO TFLite Detection — Streamlit App")

if not model_loaded:
    st.error("Model failed to load. Cannot continue.")
    st.stop()

mode = st.selectbox("Select Mode", ["Upload Image", "Live Webcam"])

# =====================================================
# MODE 1 — Upload Image
# =====================================================
if mode == "Upload Image":
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        frame = np.array(img)

        st.image(frame, caption="Original Image")

        if st.button("Run Detection"):
            try:
                boxes, scores, classes = run_tflite_inference(frame)

                frame2 = frame.copy()
                draw_boxes(frame2, boxes, scores, classes, threshold=CONF_THRESHOLD)

                st.image(frame2, caption="Detections")
            except Exception as ex:
                st.error(f"Error running inference: {ex}")
                st.code(traceback.format_exc())

# =====================================================
# MODE 2 — Live Webcam (Streamlit-safe)
# =====================================================
if mode == "Live Webcam":

    if "run_cam" not in st.session_state:
        st.session_state.run_cam = False

    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")

    if start_button:
        st.session_state.run_cam = True
    if stop_button:
        st.session_state.run_cam = False

    cam = cv2.VideoCapture(0)
    stframe = st.empty()

    while st.session_state.run_cam:
        ret, frame = cam.read()
        if not ret:
            st.error("Camera not found.")
            break

        try:
            boxes, scores, classes = run_tflite_inference(frame)
            draw_boxes(frame, boxes, scores, classes, threshold=CONF_THRESHOLD)
        except:
            pass

        stframe.image(frame, channels="BGR")

    cam.release()
