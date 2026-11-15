
import cv2
import mediapipe as mp

# ==========================================
# 1. Initialize MediaPipe Pose
# ==========================================
print("Rishabh, 23bai70136")
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing utility
mp_draw = mp.solutions.drawing_utils


# ==========================================
# 2. Open Webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


# ==========================================
# 3. Start Real-Time Detection
# ==========================================

while True:

    # Capture frame
    success, img = cap.read()

    if not success:
        print("Error: Could not read frame.")
        break

    # Get image dimensions
    h, w, c = img.shape

    # ==========================================
    # 4. Convert BGR → RGB
    # ==========================================

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ==========================================
    # 5. Process Image Using MediaPipe
    # ==========================================

    results = pose.process(imgRGB)

    # ==========================================
    # 6. Check if Pose is Detected
    # ==========================================

    if results.pose_landmarks:

        # ==========================================
        # Draw Landmarks and Skeleton
        # ==========================================

        mp_draw.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # ==========================================
        # Display All 33 Landmark Coordinates
        # ==========================================

        for id, landmark in enumerate(results.pose_landmarks.landmark):

            # Convert normalized coordinates
            # into pixel coordinates
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            # Print coordinates in terminal
            print(
                f"Landmark {id}: "
                f"x={x}, y={y}, "
                f"z={landmark.z:.2f}"
            )

        # ==========================================
        # Display Number of Landmarks
        # ==========================================

        cv2.putText(
            img,
            "Pose Detected - 33 Landmarks",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    else:

        # Display message if no person is detected
        cv2.putText(
            img,
            "No Pose Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # ==========================================
    # 7. Display Webcam Output
    # ==========================================

    cv2.imshow("Human Pose Detection", img)

    # ==========================================
    # 8. Press Q to Exit
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ==========================================
# 9. Release Resources
# ==========================================

cap.release()
cv2.destroyAllWindows()

# Close MediaPipe Pose
pose.close()

print("Pose Detection Program Closed Successfully.")
