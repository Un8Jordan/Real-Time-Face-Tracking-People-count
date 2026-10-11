import cv2
import numpy as np
import matplotlib.pyplot as plt


# Step 1: Load the image
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ret, frame = cap.read()


# Step 2: Convert to RGB for displaying with matplotlib
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Step 3: Convert to grayscale
gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Step 4: Crop the image (assume we want the region from 100:300, 200:400)
cropped_image = frame[100:300, 200:400]
cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

# Step 5: Rotate the image by 45 degrees
(h, w) = frame.shape[:2]
center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(frame, M, (w, h))
rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

# Step 6: Increase brightness by adding 50 to all pixel values
brightness_matrix = np.ones(frame.shape, dtype="uint8") * 50
brighter = cv2.add(frame, brightness_matrix)
brighter_rgb = cv2.cvtColor(brighter, cv2.COLOR_BGR2RGB)

# Show all images in one figure, ordered left-to-right and top-to-bottom.
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

images = [
    (frame_rgb, "Original Image", None),
    (gray_image, "Grayscale Image", "gray"),
    (cropped_rgb, "Cropped Region", None),
    (rotated_rgb, "Rotated Image", None),
    (brighter_rgb, "Brighter Image", None),
]

for ax, (image, title, cmap) in zip(axes.flat, images):
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")

axes.flat[-1].axis("off")  # Hide the unused sixth panel
plt.tight_layout()
plt.show()

cap.release()
cv2.destroyAllWindows()