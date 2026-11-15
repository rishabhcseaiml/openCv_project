
import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    face_detection.close()
    raise RuntimeError("Cannot open webcam")

print("MediaPipe Face Detection started.")
print("Press Q or ESC to exit.")

try:
    while True:
        success, frame = cap.read()

        if not success:
            print("Cannot read webcam frame.")
            break

        frame = cv2.flip(frame, 1)

        # OpenCV uses BGR; MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_detection.process(rgb_frame)

        face_count = 0

        if results.detections:
            face_count = len(results.detections)

            for detection in results.detections:
                mp_drawing.draw_detection(
                    frame,
                    detection
                )

        cv2.putText(
            frame,
            f"Faces detected: {face_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Experiment 9 - MediaPipe Detection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

finally:
    cap.release()
    face_detection.close()
    cv2.destroyAllWindows()
    print("MediaPipe detection stopped.")
