# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

# I had to use tensorflow 2.10 due to technical problems.
# this has led me down a rabbit hole of reinstalling different versions of different things just to run this project


# CONFIG
# testing small amount just so it doesn't take 10 billion years to run this
# also apart of Step 1, but moved here for simplicity
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 32
NUM_CLASSES = 3             # crack, missing head and paint off
EPOCHS = 20                 # CHANGE THIS WHEN TESTING

# file paths
TRAIN_DIR = r'C:\Users\baotr\Documents\GitHub\AER850-Project2-BaoTran-Submission\Data\train'
VAL_DIR = r'C:\Users\baotr\Documents\GitHub\AER850-Project2-BaoTran-Submission\Data\valid'
TEST_DIR = r'C:\Users\baotr\Documents\GitHub\AER850-Project2-BaoTran-Submission\Data\test'

############################################################################################
###################### STEP 1: DATA PROCESSING AND AUGMENTATION ############################
############################################################################################

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# generating batches for validation
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical' # Required for multi-class classification
)

# map class indices to their names for later use
class_names = list(train_generator.class_indices.keys())
print(f"Detected Class Names: {class_names}")

############################################################################################
###################### STEP 2: DATA PROCESSING AND AUGMENTATION ############################
############################################################################################

# First Conv Block
model = Sequential([
    
    # First Conv Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D(2, 2),

    # Second Conv Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Third Conv Block (Adding complexity, as discussed in the tutorial)
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Flatten and Dense layers
    Flatten(),
    Dropout(0.5), # Regularization to prevent overfitting
    Dense(512, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax') # Softmax output for multi-class probability
])

############################################################################################
############# STEP 2: HYPERPARAMETER TUNING AND MODEL COMPLIATION #########################
############################################################################################

model.compile(
    loss='categorical_crossentropy', # Appropriate loss function for multi-class one-hot encoded labels
    optimizer='adam', # A good default optimizer
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()