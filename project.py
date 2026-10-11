import os

import cv2

output_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output_images",
)

# Open the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

images = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read a frame from the webcam.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Crop the frame
    cropped = frame[100:300, 200:400]

    # Rotate the frame 45 degrees
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(frame, matrix, (w, h))

    # Increase the brightness
    brightened = cv2.add(frame, 50)

    images = {
        "grayscale.jpg": gray,
        "cropped.jpg": cropped,
        "rotated.jpg": rotated,
        "brightened.jpg": brightened,
    }

    # Display the webcam and processed images
    cv2.imshow("Webcam", frame)
    for filename, image in images.items():
        cv2.imshow(filename.split(".")[0].title(), image)

    # Press q to close the windows
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if images is not None:
    os.makedirs(output_dir, exist_ok=True)
    for filename, image in images.items():
        cv2.imwrite(os.path.join(output_dir, filename), image)
    print("Success! All images have been saved to the 'output_images' folder.")
