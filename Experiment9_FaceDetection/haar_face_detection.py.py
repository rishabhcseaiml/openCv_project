
import cv2
import os
import urllib.request

# ==========================================
# EXPERIMENT 9: HAAR CASCADE FACE DETECTION
# ==========================================

# Try to locate the Haar Cascade XML bundled with OpenCV
cascade_path = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)

# If the XML file is missing, download it
if not os.path.isfile(cascade_path):
    print("Haar Cascade XML not found. Downloading...")

    cascade_url = (
        "https://raw.githubusercontent.com/opencv/opencv/"
        "4.x/data/haarcascades/"
        "haarcascade_frontalface_default.xml"
    )

    cascade_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "haarcascade_frontalface_default.xml"
    )

    try:
        urllib.request.urlretrieve(cascade_url, cascade_path)
        print("Haar Cascade XML downloaded successfully.")
    except Exception as error:
        raise RuntimeError(
            "Could not download Haar Cascade XML. "
            "Check your internet connection."
        ) from error

# Load the Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise RuntimeError(
        f"Failed to load Haar Cascade XML: {cascade_path}"
    )

# ==========================================
# OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Cannot open webcam. Check camera permissions or connection."
    )

print("Haar Cascade Face Detection Started")
print("Press Q or ESC to exit.")

# ==========================================
# FACE DETECTION
# ==========================================

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Unable to read frame from webcam.")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Face Detected",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Show face count
        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("Experiment 9 - Haar Cascade Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Face detection stopped.")
