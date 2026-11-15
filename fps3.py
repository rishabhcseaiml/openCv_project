import cv2
import time

# Open the video
video = cv2.VideoCapture("video.mp4")

# Check whether the video opened successfully
if not video.isOpened():
    print("Error: Cannot open video")
    exit()

# Get original FPS
video_fps = video.get(cv2.CAP_PROP_FPS)

print("Original Video FPS:", video_fps)

# Store initial time
prev_time = time.time()

while True:

    # Read a frame
    ret, frame = video.read()

    # Stop when video ends
    if not ret:
        break

    # Current time
    current_time = time.time()

    # Processing FPS
    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # Display original FPS
    cv2.putText(
        frame,
        f"Video FPS: {video_fps:.2f}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    # Display processing FPS
    cv2.putText(
        frame,
        f"Current FPS: {fps:.2f}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    # Show frame
    cv2.imshow("Video", frame)

    # Exit on pressing q
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()