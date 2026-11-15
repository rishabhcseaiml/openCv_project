import cv2
import mediapipe as mp

# ==========================================
# INITIALIZE MEDIAPIPE
# ==========================================

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Face Detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,       # 468 landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:

    success, img = cap.read()

    if not success:
        break

    # Mirror webcam
    img = cv2.flip(img, 1)

    h, w, c = img.shape

    # BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ==========================================
    # FACE DETECTION
    # ==========================================

    detection_results = face_detection.process(img_rgb)

    if detection_results.detections:

        for detection in detection_results.detections:

            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Keep inside image
            x = max(0, x)
            y = max(0, y)

            # Green bounding box
            cv2.rectangle(
                img,
                (x, y),
                (x + bw, y + bh),
                (0, 255, 0),
                2
            )

    # ==========================================
    # FACE MESH
    # ==========================================

    mesh_results = face_mesh.process(img_rgb)

    if mesh_results.multi_face_landmarks:

        for face_landmarks in mesh_results.multi_face_landmarks:

            # Draw ONLY the 468 landmark points
            # No connecting lines

            for landmark in face_landmarks.landmark:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                if 0 <= x < w and 0 <= y < h:

                    cv2.circle(
                        img,
                        (x, y),
                        1,
                        (0, 255, 255),
                        -1
                    )

            # ==================================
            # DISPLAY INFORMATION
            # ==================================

            cv2.putText(
                img,
                "Face Mesh",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                img,
                "468 Landmarks",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

    # ==========================================
    # DISPLAY
    # ==========================================

    cv2.imshow(
        "Face Detection and Face Mesh",
        img
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()

face_detection.close()
face_mesh.close()