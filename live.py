import cv2
import numpy as np
import tensorflow as tf
import time
import random

# ==============================
# CONFIGURATION
# ==============================

MODEL_PATH = "best_float16.tflite"
CONF_THRESHOLD = 0.5

# Webcam: 0 = laptop webcam
# DroidCam example: "http://192.168.1.101:8080/video"
CAP_SOURCE = 0

# Class names (replace with your trained labels)
CLASS_NAMES = ["Pothole", "Crack", "Faded Lane"]  # add more if needed

# ==============================
# LOAD TFLITE MODEL
# ==============================

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite Model Loaded Successfully!")
print("Input Shape:", input_details[0]['shape'])
print("Output details:", output_details)

# ==============================
# HELPER FUNCTIONS
# ==============================

def preprocess(frame):
    h, w = input_details[0]['shape'][1:3]
    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def generate_colors(num_classes):
    random.seed(42)
    colors = []
    for _ in range(num_classes):
        colors.append(tuple([random.randint(0,255) for _ in range(3)]))
    return colors

COLORS = generate_colors(len(CLASS_NAMES))

def draw_boxes(frame, boxes, scores, classes, threshold=0.5):
    h, w, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] >= threshold:
            xmin, ymin, xmax, ymax = boxes[i]
            xmin, xmax = int(xmin * w), int(xmax * w)
            ymin, ymax = int(ymin * h), int(ymax * h)

            class_id = int(classes[i])
            color = COLORS[class_id % len(COLORS)]
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"{class_name}: {scores[i]:.2f}"
            cv2.putText(frame, label, (xmin, ymin-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ==============================
# LANE DETECTION FUNCTION
# ==============================

def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Threshold to detect bright white lines (lane markings)
    _, lane_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # OR use Canny edges: lane_mask = cv2.Canny(gray, 100, 200)
    # Convert to 3-channel for overlay
    lane_colored = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
    # Overlay on original frame
    combined = cv2.addWeighted(frame, 0.8, lane_colored, 0.2, 0)
    return combined

# ==============================
# START CAMERA
# ==============================

cap = cv2.VideoCapture(CAP_SOURCE)
if not cap.isOpened():
    print(" Could not open webcam or DroidCam. Check CAP_SOURCE.")
    exit()

print("Starting live detection. Press 'Q' to quit...")
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera.")
        break

    # Detect lane markings and overlay
    frame_lanes = detect_lanes(frame)

    # TFLite prediction
    input_image = preprocess(frame_lanes)
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]['index'])[0]  # [N,6]
    if predictions.size > 0:
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        classes = predictions[:, 5]
        draw_boxes(frame_lanes, boxes, scores, classes, CONF_THRESHOLD)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame_lanes, f"FPS: {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    # Show frame
    cv2.imshow("TFLite Live Detection + Lane Markings", frame_lanes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Live detection stopped.")