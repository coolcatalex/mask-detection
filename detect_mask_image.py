# Import necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

# Argument parser for image path and models
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load the face detector model
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the mask detector model
maskNet = load_model(args["model"])

# Load the input image and create a blob
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Detect faces
faceNet.setInput(blob)
detections = faceNet.forward()
faces, locs, preds = [], [], []

# Process each detection
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > args["confidence"]:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        # Prepare face for mask prediction
        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        faces.append(face)
        locs.append((startX, startY, endX, endY))

# Perform mask prediction if faces are detected
if len(faces) > 0:
    faces = np.array(faces, dtype="float32")
    preds = maskNet.predict(faces, batch_size=32)

# Display results
for (box, pred) in zip(locs, preds):
    (startX, startY, endX, endY) = box
    (mask, withoutMask) = pred
    label = "Mask" if mask > withoutMask else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    label = f"{label}: {max(mask, withoutMask) * 100:.2f}%"

    # Display label and bounding box
    cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# Show final image
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
