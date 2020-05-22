import argparse
import logging
import os

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Set logging level
logging.basicConfig(level=logging.INFO)


def parse_arguments() -> dict:
    """
    Parse arguments in terminal and collects them in dict.
    Key is argument name and value is the value for that argument.

    Returns:
        __dict__ attribute of a given object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image",
                        required=True,
                        help="path to input image")
    parser.add_argument("-f", "--face",
                        type=str,
                        default="face_detector",
                        help="path to face detector model directory")
    parser.add_argument("-m", "--model",
                        type=str,
                        default="mask_detector.model",
                        help="path to trained face mask detector model")
    parser.add_argument("-c", "--confidence",
                        type=float,
                        default=0.5,
                        help="minimum probability to filter out weak detections")

    arguments = vars(parser.parse_args())
    return arguments


# Call arguments parser
args = parse_arguments()


# Load the face detector model from disk
logging.info("Loading Face Detector Model...")
prototxtPath = os.path.sep.join([args.get("face"), "deploy.prototxt"])
weightsPath = os.path.sep.join([args.get("face"),
                                "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)


# Load the face mask detector model from disk
logging.info("Loading Face Mask Detector Model...")
model = load_model(args.get("model"))


# Load the input image from disk, clone it, and grab the image spatial dimensions
image = cv2.imread(args.get("image"))
orig = image.copy()
(h, w) = image.shape[:2]


# Construct a blob from the image
# Resize to (300x300) pixels and perform mean subtraction
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                             (104.0, 177.0, 123.0))


# Pass the blob through the network and obtain the face detections
logging.info("Computing Face Detection...")
net.setInput(blob)
detections = net.forward()


# Iterate over the detections
for i in range(0, detections.shape[2]):
    # Extract the confidence (i.e., probability) associated with the detection
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
    if confidence > args.get("confidence"):
        # Compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Ensure the bounding boxes fall within the dimensions of the frame
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        # Extract the face Region of Interest (ROI), convert it from BGR to RGB channel,
        # resize it to 224x224, and pre-process.
        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # PREDICTION
        # Pass the face through the model to determine if the face has a mask or not
        (mask, withoutMask) = model.predict(face)[0]

        # Determine the class label and color we'll use to draw the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        # Green color for mask, red color for without_mask
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Display the label and bounding box rectangle on the output frame
        cv2.putText(image,
                    label,
                    (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 2)
        cv2.rectangle(image,
                      (startX, startY),
                      (endX, endY),
                      color, 2)


# Show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
