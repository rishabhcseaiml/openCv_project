import os

print("Current directory:", os.getcwd())
import cv2

# Read Image
img = cv2.imread(r"C:\Users\risha\OneDrive\Desktop\openCv_project\ImageProcessingLab\input.jpg")

# Check whether image is loaded
if img is None:
    print("Image not found!")
    exit()

# Display Original Image
cv2.imshow("Original Image", img)

# Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray)

# Resize Image
resized = cv2.resize(img, (400, 300))
cv2.imshow("Resized Image", resized)

# Rotate Image
rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("Rotated Image", rotated)

# Save Gray Image
cv2.imwrite("gray_image.jpg", gray)

print("Gray image saved successfully.")

# Wait for any key
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()