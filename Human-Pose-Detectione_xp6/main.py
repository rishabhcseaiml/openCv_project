import cv2
import mediapipe as mp

# -----------------------------------
# Initialize MediaPipe Pose
# -----------------------------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Drawing utility
mp_draw = mp.solutions.drawing_utils


# -----------------------------------
# Open Webcam
# -----------------------------------

cap = cv2.VideoCapture(0)

while True:

    # Capture frame
    success, img = cap.read()

    if not success:
        print("Unable to access webcam")
        break

    # -----------------------------------
    # Convert BGR to RGB
    # -----------------------------------

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -----------------------------------
    # Process image using MediaPipe
    # -----------------------------------

    results = pose.process(imgRGB)

    # -----------------------------------
    # Detect and Draw Landmarks
    # -----------------------------------

    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # -----------------------------------
    # Display Output
    # -----------------------------------

    cv2.imshow("Human Pose Detection", img)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# -----------------------------------
# Release Resources
# -----------------------------------

cap.release()
cv2.destroyAllWindows()