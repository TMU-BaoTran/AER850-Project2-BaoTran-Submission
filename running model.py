import os
import numpy as np
import tensorflow as tf
from google.colab import drive
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

import matplotlib.pyplot as plt

############################################################################################
########## STEP 5: NEURAL NETWORK ARCHITECTURE DESIGN/HYPER PARAMETER ANALYSIS #############
############################################################################################

print("Mounting Google Drive...")
drive.mount('/content/gdrive')

# redefine paths for test
PROJECT_ROOT = '/content/gdrive/MyDrive/Colab Notebooks/Data'
MODEL_PATH = os.path.join(PROJECT_ROOT, 'project2_aircraft_defect.h5')
TEST_DIR = os.path.join(PROJECT_ROOT, 'test') 

# redefine global configuration
IMAGE_SIZE = (500, 500) 
CLASS_LABELS = ['crack', 'missing-head', 'paint-off'] 

# loading model
print("\nLoading trained model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

### make picture

def predict_test_image(folder_name, full_image_path, model):
    """Loads a single image, preprocesses it, makes a prediction, and plots the results (Figure 3)."""

    # Construct the full file path by simply using the provided argument
    image_path = full_image_path

    # Extract the file name for display/printing
    image_file_name = os.path.basename(full_image_path)

    # The 'True' label is derived from the key in the dictionary (e.g., 'crack')
    true_label = folder_name.replace('-', ' ').title()

    print(f"\n--- Testing Image: {image_file_name} ---")

    if not os.path.exists(image_path):
        print(f"File not found at: {image_path}")
        return

    # load and preprocess

    # Load image only once for both display and array conversion
    img_display = load_img(image_path, target_size=IMAGE_SIZE)

    # show image (Optional: small plot for confirmation)
    plt.figure(figsize=(4, 4))
    plt.imshow(img_display)
    plt.title(f"Test Image: {image_file_name}")
    plt.axis('off') # Hide axis ticks
    plt.show()

    # Convert to array
    img_array = img_to_array(img_display)

    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # make Prediction
    predictions = model.predict(img_array, verbose=0)[0]

    # Get the predicted class (the one with the maximum probability)
    predicted_index = np.argmax(predictions)
    predicted_label = CLASS_LABELS[predicted_index].replace('-', ' ').title()

    # Create the figure for the final output
    plt.figure(figsize=(6, 8))

    # Display the image
    plt.imshow(img_display)
    plt.axis('off')

    # add text overlay
    text_y = 0.55 # Controls vertical placement of the text

    # Find the maximum probability to highlight it
    max_prob = predictions.max()

    # Loop through all 3 probabilities to display and highlight them
    for label, prob in zip(CLASS_LABELS, predictions):
        # Set appearance based on confidence to highlight the prediction
        if prob == max_prob:
            text_color = 'yellow' # Highlight color
            font_size = 16
        else:
            text_color = 'lime' # Standard color
            font_size = 14

        formatted_label = label.replace('-', ' ').title()
        text_output = f"{formatted_label}: {prob*100:.1f}%"

        # Add the probability text onto the image
        plt.text(50, # X position
               int(text_y * IMAGE_SIZE[0]), # Y position
               text_output,
               fontsize=font_size,
               color=text_color,
               fontweight='bold')

        text_y += 0.05 # Move down for the next line

    # Add True and Predicted Labels as the plot title
    title_text = f"True Crack Classification Label: {true_label}"
    subtitle_text = f"Predicted Crack Classification Label: {predicted_label}"

    plt.title(f"{title_text}\n{subtitle_text}\n", fontsize=16, loc='left', pad=20)
    plt.tight_layout()
    plt.show()


# DEFINING PATH
test_cases = {
    'crack': '/content/gdrive/MyDrive/Colab Notebooks/Data/test/crack/test_crack.jpg',
    'missing-head': '/content/gdrive/MyDrive/Colab Notebooks/Data/test/missing-head/test_missinghead.jpg',
    'paint-off': '/content/gdrive/MyDrive/Colab Notebooks/Data/test/paint-off/test_paintoff.jpg'
}

for folder, file in test_cases.items():
    predict_test_image(folder, file, model)