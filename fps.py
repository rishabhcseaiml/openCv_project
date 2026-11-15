import cv2
import time
import numpy as np

# Open the video
video = cv2.VideoCapture(r"C:\Users\risha\OneDrive\Desktop\openCv_project\video.mp4")

# Check if the video opened successfully
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Get original video FPS
video_fps = video.get(cv2.CAP_PROP_FPS)
print("Original Video FPS:", video_fps)

prev_time = time.time()

while True:

    # Read one frame
    ret, frame = video.read()

    # Stop if video ends
    if not ret:
        print("Video Finished")
        break

    # Calculate current FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # -----------------------------
    # Create White Header
    # -----------------------------
    header_height = 80

    header = np.ones((header_height, frame.shape[1], 3), dtype=np.uint8) * 255

    # Name
    cv2.putText(
        header,
        "Name : RISHABH RAIKWAR",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    # UID
    cv2.putText(
        header,
        "UID : 23BAI70136",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    # Display FPS on the frame
    cv2.putText(
        frame,
        f"FPS : {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # Display Original Video FPS
    cv2.putText(
        frame,
        f"Original FPS : {video_fps:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Combine Header + Frame
    output = np.vstack((header, frame))

    # Display Video
    cv2.imshow("Video FPS", output)

    # Press q to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()