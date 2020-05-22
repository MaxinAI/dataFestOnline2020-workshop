import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

# Set logging level
logging.basicConfig(level=logging.INFO)


# Define the initial learning rate, number of epochs and batch size
INIT_LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 32


def parse_arguments() -> dict:
    """
    Parse arguments in terminal and collects them in dict.
    Key is argument name and value is the value for that argument.

    Returns:
        __dict__ attribute of a given object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset",
                        required=True,
                        help="path to input dataset")
    parser.add_argument("-p", "--plot",
                        type=str,
                        default="plot.png",
                        help="path to output loss/accuracy plot")
    parser.add_argument("-m", "--model",
                        type=str,
                        default="mask_detector.model",
                        help="path to output face mask detector model")

    arguments = vars(parser.parse_args())
    return arguments


# Call arguments parser
args = parse_arguments()


# Grab the list of images in our data-set directory, then initialize
# the list of images and class images.
logging.info("Loading Images...")
imagePaths = list(paths.list_images(args.get("dataset")))
data = []
labels = []


# Iterate over the image paths
for imagePath in imagePaths:
    # Extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # Resize images to (224x224) pixels
    image = load_img(imagePath, target_size=(224, 224))
    # Convert into array
    image = img_to_array(image)
    # Scale pixel intensities to [-1, 1]
    image = preprocess_input(image)

    # Update pre-processed image and its label into list
    data.append(image)
    labels.append(label)


# Convert the data and labels to Numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)


# One-Hot encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


# Split the data into training and testing set
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels,
                                                      test_size=0.20,
                                                      stratify=labels,
                                                      random_state=42)


# Construct the training image generator for data augmentation
aug = ImageDataGenerator(
    # Degree range for random rotations
    rotation_range=20,
    # Range for random zoom
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    # Shear angle in counter-clockwise direction in degrees
    shear_range=0.15,
    # Randomly flip inputs horizontally
    horizontal_flip=True,
    fill_mode="nearest")


# Load the MobileNetV2 with pre-trained ImageNet weights.
# Ensuring the head FC layer sets are left off
baseModel = MobileNetV2(weights="imagenet",
                        include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))


# Construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)


# Place the head FC model on top of the base model
# (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)


# Loop over all layers in the base model and freeze them so they will NOT be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False


# Compile the Model
logging.info("Compiling the Model...")
# Optimizer function
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])


# Train the head of the network
logging.info("Training Head...")
H = model.fit(
    aug.flow(X_train,
             Y_train,
             batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    validation_data=(X_test, Y_test),
    validation_steps=len(X_test) // BATCH_SIZE,
    epochs=EPOCHS)


# Make predictions on the testing set
logging.info("Evaluating the Model")
predIdxs = model.predict(X_test, batch_size=BATCH_SIZE)


# For each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)


# Show classification report
print(classification_report(Y_test.argmax(axis=1),
                            predIdxs,
                            target_names=lb.classes_))


# Serialize the model to disk
logging.info("Saving Mask Detector Model...")
model.save(args.get("model"), save_format="h5")


# Plot the training loss and accuracy
N = EPOCHS
# Set the plotting style
plt.style.use("ggplot")
# Define the figure object
plt.figure()

plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args.get("plot"))
