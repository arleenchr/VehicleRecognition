import os
import numpy as np
from PIL import Image

# Assuming preprocess_image returns features and hog_image
def load_dataset(dataset_path):
    features = []
    labels = []
    
    for label in os.listdir(dataset_path):  # Loop through each folder (Ambulance, Car, Truck)
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                try:
                    image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
                    image = image.resize((1200, 800), Image.ANTIALIAS)  # Resize image
                    
                    # Extract features using preprocess_image
                    feature_vector = np.array(image).flatten()
                    features.append(feature_vector)
                    labels.append(label)  # Label is the folder name
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                
    # Print total number of images processed
    for label in set(labels):
        print(f"Number of images for label '{label}': {labels.count(label)}")
    
    return np.array(features), np.array(labels)