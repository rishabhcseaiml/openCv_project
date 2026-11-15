import cv2
from math import sqrt

# --------------------------------------------------
# Read image
# --------------------------------------------------

img = cv2.imread(r"C:\Users\risha\OneDrive\Desktop\openCv_project\Experiment8_CameraCalibration\ChatGPT Image Sep 30, 2026, 05_43_51 PM.png")

if img is None:
    print("Error: pen.jpg not found!")
    exit()

# --------------------------------------------------
# Resize only for display
# --------------------------------------------------

scale = 0.25

display_img = cv2.resize(
    img,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_AREA
)

# ==================================================
# STEP 1: SELECT RULER POINTS
# ==================================================

ruler_points = []

print("\n======================================")
print("STEP 1: RULER CALIBRATION")
print("======================================")
print("Click TWO points on the ruler.")
print("For example:")
print("Point 1 -> 0 cm")
print("Point 2 -> 5 cm")
print("Press ENTER after selecting both points.")

def ruler_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(ruler_points) < 2:

            # Convert displayed coordinates
            # to original image coordinates
            original_x = int(x / scale)
            original_y = int(y / scale)

            ruler_points.append(
                (original_x, original_y)
            )

            # Draw point on displayed image
            cv2.circle(
                display_img,
                (x, y),
                6,
                (0, 255, 0),
                -1
            )

            print(
                "Ruler point:",
                original_x,
                original_y
            )


cv2.namedWindow("Ruler Calibration")
cv2.setMouseCallback(
    "Ruler Calibration",
    ruler_callback
)

while True:

    cv2.imshow(
        "Ruler Calibration",
        display_img
    )

    key = cv2.waitKey(1) & 0xFF

    # ENTER
    if key == 13:
        break

    # ESC
    if key == 27:
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()


# --------------------------------------------------
# Check ruler points
# --------------------------------------------------

if len(ruler_points) != 2:

    print("ERROR: Select exactly 2 ruler points.")
    exit()


# --------------------------------------------------
# Enter known distance
# --------------------------------------------------

known_length_mm = float(
    input(
        "\nEnter distance between ruler points (mm): "
    )
)


# --------------------------------------------------
# Calculate calibration
# --------------------------------------------------

ruler_pt1 = ruler_points[0]
ruler_pt2 = ruler_points[1]

reference_pixels = sqrt(
    (ruler_pt2[0] - ruler_pt1[0]) ** 2 +
    (ruler_pt2[1] - ruler_pt1[1]) ** 2
)

pixels_per_mm = (
    reference_pixels /
    known_length_mm
)

print("\nCalibration Result")
print("--------------------------")
print("Reference pixels =", reference_pixels)
print("Known distance    =", known_length_mm, "mm")
print("Pixels per mm     =", pixels_per_mm)


# ==================================================
# STEP 2: SELECT PEN POINTS
# ==================================================

pen_points = []

# Create fresh image
pen_display = cv2.resize(
    img,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_AREA
)

print("\n======================================")
print("STEP 2: PEN MEASUREMENT")
print("======================================")
print("Click the START of the pen.")
print("Then click the END of the pen.")
print("Press ENTER after selecting both points.")


def pen_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(pen_points) < 2:

            # Convert display coordinate
            # to original coordinate
            original_x = int(x / scale)
            original_y = int(y / scale)

            pen_points.append(
                (original_x, original_y)
            )

            # Draw red point
            cv2.circle(
                pen_display,
                (x, y),
                6,
                (0, 0, 255),
                -1
            )

            print(
                "Pen point:",
                original_x,
                original_y
            )


cv2.namedWindow("Select Pen")
cv2.setMouseCallback(
    "Select Pen",
    pen_callback
)

while True:

    cv2.imshow(
        "Select Pen",
        pen_display
    )

    key = cv2.waitKey(1) & 0xFF

    if key == 13:
        break

    if key == 27:
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()


# --------------------------------------------------
# Check pen points
# --------------------------------------------------

if len(pen_points) != 2:

    print("ERROR: Select exactly 2 pen points.")
    exit()


# --------------------------------------------------
# Calculate pen length
# --------------------------------------------------

pen_pt1 = pen_points[0]
pen_pt2 = pen_points[1]

pen_pixels = sqrt(
    (pen_pt2[0] - pen_pt1[0]) ** 2 +
    (pen_pt2[1] - pen_pt1[1]) ** 2
)

pen_length_mm = (
    pen_pixels /
    pixels_per_mm
)


# ==================================================
# STEP 3: GROUND TRUTH
# ==================================================

actual_length = float(
    input(
        "\nEnter actual pen length measured with ruler (mm): "
    )
)

error = abs(
    actual_length -
    pen_length_mm
)

percentage_error = (
    error /
    actual_length
) * 100


# ==================================================
# FINAL RESULT
# ==================================================

print("\n======================================")
print("           FINAL RESULT")
print("======================================")

print(
    "Pen length in pixels :",
    round(pen_pixels, 2)
)

print(
    "Estimated length     :",
    round(pen_length_mm, 2),
    "mm"
)

print(
    "Actual length        :",
    actual_length,
    "mm"
)

print(
    "Error                :",
    round(error, 2),
    "mm"
)

print(
    "Percentage error     :",
    round(percentage_error, 2),
    "%"
)

print("======================================")


# --------------------------------------------------
# Show final image
# --------------------------------------------------

cv2.imshow(
    "Final Pen Measurement",
    pen_display
)

cv2.waitKey(0)
cv2.destroyAllWindows()