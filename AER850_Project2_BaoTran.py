# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/gdrive')

PROJECT_ROOT = '/content/gdrive/MyDrive/Colab Notebooks/Data'

# Define the training and validation directories
TRAIN_DIR = os.path.join(PROJECT_ROOT, 'train')
VAL_DIR = os.path.join(PROJECT_ROOT, 'valid')

############################################################################################
###################### STEP 1: DATA PROCESSING AND AUGMENTATION ############################
############################################################################################

##### CONFIG #######
IMG_WIDTH, IMG_HEIGHT = 500, 500
BATCH_SIZE = 32
NUM_CLASSES = 3             # crack, missing head and paint off
EPOCHS = 30                 # CHANGE THIS WHEN TESTING

train_datagen = ImageDataGenerator(       # HYPERPARAMETER TUNING
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
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
    class_mode='categorical' 
)

# generating batches for validation
validation_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical' 
)


# map class indices to their names for later use
class_names = list(train_generator.class_indices.keys())
print(f"Detected Class Names: {class_names}")

############################################################################################
########## STEP 2 AND 3: NEURAL NETWORK ARCHITECTURE DESIGN/HYPER PARAMETER ANALYSIS #######
############################################################################################

# First Conv Block
model = Sequential([

    # First Conv Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D(2, 2),

    # Second Conv Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Third Conv Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Flatten and Dense layers
    Flatten(),
    Dropout(0.6),                                 # CHANGE IF NEEDED
    Dense(512, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax') 
])

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam',
    metrics=['accuracy']
)

print("\nModel Summary:")     # not required, only need it to see summary
model.summary()


############################################################################################
########## STEP 4: NEURAL NETWORK ARCHITECTURE DESIGN/HYPER PARAMETER ANALYSIS #############
############################################################################################

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

###### SAVE MODEL #######
model_save_path = os.path.join(PROJECT_ROOT, 'project2_aircraft_defect.h5')

model.save(model_save_path)
print(f"\nModel successfully saved to: {model_save_path}")
