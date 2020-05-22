import argparse
import logging
import os
import time

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Set logging level
logging.basicConfig(level=logging.INFO)


def detect_and_predict_mask(frame, faceNet, maskNet) -> tuple:
    """
    Detects faces and predicts if person wear mask or not

    Args:
        frame - A frame from video stream
        faceNet - The model to detect faces
        maskNet - The model to detect faces with mask

    Returns:
        Tuple of tuples. Face location coordinates and predicted coordinates
    """
    # Grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame,
                                 scalefactor=1.0,
                                 size=(300, 300),
                                 mean=(104.0, 177.0, 123.0))  # preprocessing which was used for face detection model training

    # Pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network

    # ROI
    faces = []
    # Face locations
    locs = []
    # Mask/No Mask predictions
    preds = []

    # Loop over the face detections
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

            # Extract the face ROI, convert it from BGR to RGB channel,
            # resize it to 224x224, and pre-process
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # Add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # Only make a predictions if at least one face was detected
    if len(faces) > 0:
        # For faster inference we'll make batch predictions on all
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        preds = maskNet.predict(faces)

    # Return face bounding-box location and corresponding mask/no mask prediction
    return (locs, preds)


def parse_arguments() -> dict:
    """
    Parse arguments in terminal and collects them in dict.
    Key is argument name and value is the value for that argument.

    Returns:
        __dict__ attribute of a given object
    """
    parser = argparse.ArgumentParser()
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
                        help="minimum probability to filter weak detections")

    arguments = vars(parser.parse_args())
    return arguments


# Call arguments parser
args = parse_arguments()


# Load the face detector model from disk
logging.info("Loading Face Detector Model...")
prototxtPath = os.path.sep.join([args.get("face"), "deploy.prototxt"])
weightsPath = os.path.sep.join([args.get("face"),
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


# Load the face mask detector model from disk
logging.info("Loading Face Mask Detector Model...")
maskNet = load_model(args.get("model"))


# Initialize the video stream and allow the camera sensor to warm up
logging.info("Starting Video Stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


# Loop over the frames from the video stream
while True:
    # Grab the frame from the threaded video stream and resize it 
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Detect faces in the frame and determine if they are wearing a face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # Loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):
        # Unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # Determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
