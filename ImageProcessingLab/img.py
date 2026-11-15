import cv2
import numpy as np

# -----------------------------------------
# Start Webcam
# -----------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found!")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Unable to read camera")
        break

    # Resize for processing
    frame = cv2.resize(frame, (800, 600))

    # -----------------------------------------
    # Convert to Grayscale
    # -----------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------
    # Reduce Noise
    # -----------------------------------------
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # -----------------------------------------
    # Detect Edges
    # -----------------------------------------
    edges = cv2.Canny(blur, 50, 150)

    # -----------------------------------------
    # Find Contours
    # -----------------------------------------
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    id_card = None

    # -----------------------------------------
    # Check Every Contour
    # -----------------------------------------
    for contour in contours:

        # Calculate contour area
        area = cv2.contourArea(contour)

        # Ignore small objects
        if area < 20000:
            continue

        # Approximate contour
        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        # ID card should approximately have 4 corners
        if len(approx) == 4:

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)

            aspect_ratio = w / float(h)

            # ID card is generally rectangular
            if 1.2 < aspect_ratio < 2.0:

                id_card = approx

                break

    # -----------------------------------------
    # If ID Card Found
    # -----------------------------------------
    if id_card is not None:

        # Draw green border
        cv2.drawContours(
            frame,
            [id_card],
            -1,
            (0, 255, 0),
            4
        )

        # Get corner points
        points = id_card.reshape(4, 2)

        # Sort points
        rect = np.zeros((4, 2), dtype=np.float32)

        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)

        rect[0] = points[np.argmin(s)]       # Top-left
        rect[2] = points[np.argmax(s)]       # Bottom-right
        rect[1] = points[np.argmin(diff)]    # Top-right
        rect[3] = points[np.argmax(diff)]    # Bottom-left

        # -----------------------------------------
        # Destination size
        # -----------------------------------------
        width = 600
        height = 400

        destination = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])

        # -----------------------------------------
        # Perspective Transformation
        # -----------------------------------------
        matrix = cv2.getPerspectiveTransform(
            rect,
            destination
        )

        card = cv2.warpPerspective(
            frame,
            matrix,
            (width, height)
        )

        # -----------------------------------------
        # Display Detection Text
        # -----------------------------------------
        cv2.putText(
            frame,
            "ID CARD DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        # Display corrected ID card
        cv2.imshow(
            "Detected ID Card",
            card
        )

    else:

        cv2.putText(
            frame,
            "Searching for ID Card...",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # -----------------------------------------
    # Display Webcam
    # -----------------------------------------
    cv2.imshow(
        "Live ID Card Detection",
        frame
    )

    # -----------------------------------------
    # Press Q to Exit
    # -----------------------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break


# -----------------------------------------
# Release Resources
# -----------------------------------------
cap.release()
cv2.destroyAllWindows()