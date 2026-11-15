import cv2
import glob
import os

# Student Details
NAME = "Adarsh K
UID = "23BAI70091"

# Folder containing images
folder = "C:/Users/risha/OneDrive/Desktop/openCv_project"

# Get all PNG, JPG and JPEG images
image_files = (
    glob.glob(folder + "/*.png")
    + glob.glob(folder + "/*.jpg")
    + glob.glob(folder + "/*.jpeg")
)

# Process each image
for image_path in image_files:

    print("Processing:", os.path.basename(image_path))

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print("Could not read:", image_path)
        continue

    output = image.copy()

    # -----------------------------
    # Convert to grayscale
    # -----------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -----------------------------
    # Gaussian Blur
    # -----------------------------
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # -----------------------------
    # Thresholding
    # -----------------------------
    _, thresh = cv2.threshold(
        blur,
        60,
        255,
        cv2.THRESH_BINARY
    )

    # -----------------------------
    # Find contours
    # -----------------------------
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    shape_count = 0

    # -----------------------------
    # Process each contour
    # -----------------------------
    for cnt in contours:

        # Calculate contour area
        area = cv2.contourArea(cnt)

        # Ignore small noise
        if area < 500:
            continue

        # Calculate perimeter
        perimeter = cv2.arcLength(cnt, True)

        # Approximate contour
        approx = cv2.approxPolyDP(
            cnt,
            0.04 * perimeter,
            True
        )

        # Count vertices
        vertices = len(approx)

        # -----------------------------
        # Identify shape
        # -----------------------------
        if vertices == 3:
            shape = "Triangle"

        elif vertices == 4:

            x, y, w, h = cv2.boundingRect(approx)

            if abs(w - h) <= 10:
                shape = "Square"
            else:
                shape = "Rectangle"

        elif vertices == 5:
            shape = "Pentagon"

        else:
            shape = "Circle"

        shape_count += 1

        # -----------------------------
        # Draw contour
        # -----------------------------
        cv2.drawContours(
            output,
            [approx],
            -1,
            (0, 255, 0),
            2
        )

        # -----------------------------
        # Bounding box
        # -----------------------------
        x, y, w, h = cv2.boundingRect(approx)

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # -----------------------------
        # Display shape name
        # -----------------------------
        cv2.putText(
            output,
            shape,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    # -----------------------------
    # Display Student Name
    # -----------------------------
    cv2.putText(
        output,
        "Name: " + NAME,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # -----------------------------
    # Display UID
    # -----------------------------
    cv2.putText(
        output,
        "UID: " + UID,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # -----------------------------
    # Display total shapes
    # -----------------------------
    cv2.putText(
        output,
        "Total Shapes: " + str(shape_count),
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # -----------------------------
    # Display output
    # -----------------------------
    window_name = "Detected Shapes - " + os.path.basename(image_path)

    cv2.imshow(window_name, output)

    # Wait for key
    cv2.waitKey(0)

    # Close window
    cv2.destroyAllWindows()

print("All images processed successfully.")