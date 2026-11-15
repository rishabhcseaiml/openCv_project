import cv2
import numpy as np

# ---------------------------------
# Open Webcam
# ---------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found!")
    exit()

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ---------------------------------
# Perspective Transformation Function
# ---------------------------------
def perspective_transform(frame):

    h, w = frame.shape[:2]

    # Four points selected from original frame
    src_points = np.float32([
        [100, 100],          # Top-left
        [w - 100, 100],      # Top-right
        [w - 100, h - 100],  # Bottom-right
        [100, h - 100]       # Bottom-left
    ])

    # Destination points
    dst_points = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])

    # Calculate transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply perspective transformation
    transformed = cv2.warpPerspective(
        frame,
        matrix,
        (w, h)
    )

    return transformed


# ---------------------------------
# Live Video Loop
# ---------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("Unable to read camera!")
        break

    # ---------------------------------
    # Original
    # ---------------------------------
    original = frame.copy()

    # ---------------------------------
    # Grayscale
    # ---------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale to BGR
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ---------------------------------
    # Resize
    # ---------------------------------
    resized = cv2.resize(frame, (400, 300))

    # ---------------------------------
    # Rotate
    # ---------------------------------
    rotated = cv2.rotate(
        frame,
        cv2.ROTATE_90_CLOCKWISE
    )

    # ---------------------------------
    # Perspective Transformation
    # ---------------------------------
    perspective = perspective_transform(frame)

    # ---------------------------------
    # Flip
    # ---------------------------------
    flipped = cv2.flip(frame, 1)


    # ---------------------------------
    # Draw Perspective Selection Points
    # ---------------------------------
    h, w = frame.shape[:2]

    points = [
        (100, 100),
        (w - 100, 100),
        (w - 100, h - 100),
        (100, h - 100)
    ]

    for point in points:
        cv2.circle(
            original,
            point,
            8,
            (0, 0, 255),
            -1
        )


    # ---------------------------------
    # Display
    # ---------------------------------
    cv2.imshow("Original Live Video", original)

    cv2.imshow("Grayscale", gray_bgr)

    cv2.imshow("Resized", resized)

    cv2.imshow("Rotated", rotated)

    cv2.imshow("Perspective Transformation", perspective)

    cv2.imshow("Flipped", flipped)


    # ---------------------------------
    # Press Q to Exit
    # ---------------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break


# ---------------------------------
# Release Camera
# ---------------------------------
cap.release()

cv2.destroyAllWindows()