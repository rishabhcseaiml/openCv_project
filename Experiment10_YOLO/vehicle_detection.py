
import cv2
from ultralytics import YOLO
import os

# ==========================================
# EXPERIMENT 10: YOLO VEHICLE DETECTION
# ==========================================

# Get the folder where this Python file is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load pretrained YOLO model (NOT the video file)
model = YOLO("yolov8n.pt", task="detect")

# Input video path: traffic.mp4 must be in the same folder
VIDEO_PATH = os.path.join(BASE_DIR, "traffic.mp4")

# False = use video file, True = use webcam
USE_WEBCAM = False

# COCO class IDs:
# 2 = Car, 5 = Bus, 7 = Truck
VEHICLE_CLASSES = {
    2: "Car",
    5: "Bus",
    7: "Truck"
}

CONFIDENCE_THRESHOLD = 0.40

# ==========================================
# OPEN VIDEO SOURCE
# ==========================================

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    if not os.path.isfile(VIDEO_PATH):
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}\n"
            "Make sure traffic.mp4 is in the same folder "
            "as vehicle_detection.py."
        )

    cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        "Unable to open video source. Check the video path "
        "or webcam connection."
    )

print("YOLO Vehicle Detection Started")
print("Detecting: Cars, Buses and Trucks")
print("Press Q or ESC to exit.")

# ==========================================
# PROCESS VIDEO FRAME BY FRAME
# ==========================================

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Video ended or frame could not be read.")
            break

        # Run YOLO detection on the current frame
        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        vehicle_counts = {
            "Car": 0,
            "Bus": 0,
            "Truck": 0
        }

        # Process detected objects
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])

                # Ignore objects other than cars, buses and trucks
                if class_id not in VEHICLE_CLASSES:
                    continue

                label = VEHICLE_CLASSES[class_id]
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                vehicle_counts[label] += 1

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Display vehicle label and confidence
                text = f"{label}: {confidence:.2f}"

                cv2.putText(
                    frame,
                    text,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Display vehicle counts on the frame
        cv2.putText(
            frame,
            f"Cars: {vehicle_counts['Car']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Buses: {vehicle_counts['Bus']}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Trucks: {vehicle_counts['Truck']}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Display the annotated frame
        cv2.imshow(
            "Experiment 10 - YOLO Vehicle Detection",
            frame
        )

        # Exit on Q or ESC
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Vehicle detection stopped.")
