import os
import numpy as np
from PIL import Image
from conventional.preprocessing import preprocess_image

# Assuming preprocess_image returns features and hog_image
def load_dataset(dataset_paths):
    features = []
    labels = []
    
    for dataset_path in dataset_paths:
        if not os.path.isdir(dataset_path):
            print(f"Dataset path '{dataset_path}' does not exist or is not a directory.")
            continue
        
        for label in os.listdir(dataset_path):  # Loop through each folder (Ambulance, Car, Truck)
            label_path = os.path.join(dataset_path, label)
            if os.path.isdir(label_path):
                for image_file in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_file)
                    try:
                        # Load and preprocess the image
                        image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
                        image = image.resize((256, 256))  # Resize image
                        
                        # Extract features using preprocess_image
                        feature_vector = preprocess_image(image, target_feature_size=1000)
                        features.append(feature_vector)
                        labels.append(label)  # Label is the folder name
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
    
    # Logging statistics
    print(f"Processed {len(features)} images in total.")
    for label in set(labels):
        print(f"Number of images for label '{label}': {labels.count(label)}")
    
    return np.array(features), np.array(labels)