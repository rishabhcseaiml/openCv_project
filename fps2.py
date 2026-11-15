import cv2
import time

# Open webcam
camera = cv2.VideoCapture(0)

# Check camera
if not camera.isOpened():
    print("Cannot open camera")
    exit()

# Initial time
prev_time = time.time()

while True:

    # Capture frame
    ret, frame = camera.read()

    if not ret:
        print("Cannot receive frame")
        break

    # Current time
    current_time = time.time()

    # Calculate FPS
    fps = 1 / (current_time - prev_time)

    # Update previous time
    prev_time = current_time

    # Display FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    # Display frame
    cv2.imshow("Live Camera FPS", frame)

    # Exit on q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()