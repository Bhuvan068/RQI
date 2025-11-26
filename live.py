import cv2
import numpy as np
import time
from PIL import Image
import tflite_runtime.interpreter as tflite

# -----------------------------
# LOAD TFLITE MODEL
# -----------------------------
MODEL_PATH = "best_float16.tflite"   # your model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img_height = input_details[0]['shape'][1]
img_width = input_details[0]['shape'][2]


# -----------------------------
# SET CAMERA SOURCE
# -----------------------------
# 0 = laptop webcam
# For IP Webcam use:  "http://<phone-ip>:8080/video"

USE_IP_WEBCAM = True
IP_URL = "http://192.168.1.5:8080/video"   # change to your phone’s IP

if USE_IP_WEBCAM:
    cap = cv2.VideoCapture(IP_URL)
else:
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opening...")
    exit()


# -----------------------------
# MODEL INFERENCE FUNCTION
# -----------------------------
def detect_objects(frame):
    img = cv2.resize(frame, (img_width, img_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    # YOLO output format:
    # [x1, y1, x2, y2, score, class]
    results = interpreter.get_tensor(output_details[0]['index'])
    return results[0]


# -----------------------------
# LANE HIGHLIGHT FUNCTION
# -----------------------------
def highlight_lane(frame, x1, y1, x2, y2):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)  # yellow overlay
    return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)


# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received...")
        break

    detections = detect_objects(frame)

    for det in detections:
        x1, y1, x2, y2, score, cls = det

        if score < 0.5:
            continue

        # Convert normalized coords to pixels
        h, w, _ = frame.shape
        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)

        # Lane class = 0 (change if yours is different)
        if int(cls) == 0:
            frame = highlight_lane(frame, x1, y1, x2, y2)
            cv2.putText(frame, f"Lane {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Obj {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Live Lane Detection (YOLO-TFLite + IP Camera)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
