import os
import numpy as np
from PIL import Image
from conventional.preprocess_image import preprocess_image
from constant import IMAGE_HEIGHT, IMAGE_WIDTH

# Assuming preprocess_image returns features and hog_image
def load_dataset(dataset_paths, amount_each_class: int = 200):
    features = []
    labels = []
    
    for dataset_path in dataset_paths:
        print(f'[PROCESSING DATASET] {dataset_path}')
        if not os.path.isdir(dataset_path):
            print(f"Dataset path '{dataset_path}' does not exist or is not a directory.")
            continue
        
        for label in os.listdir(dataset_path):  # Loop through each folder (Ambulance, Car, Truck)
            label_path = os.path.join(dataset_path, label)
            if os.path.isdir(label_path):
                for image_file in os.listdir(label_path)[:amount_each_class]:
                    image_path = os.path.join(label_path, image_file)
                    print(f'[EXTRACTING FEATURE] {image_path}')
                    try:
                        # Load and preprocess the image
                        image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
                        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))  # Resize image
                        
                        # Extract features using preprocess_image
                        feature_vector, _ = preprocess_image(image, target_feature_size=1000)
                        features.append(feature_vector)
                        labels.append(label)  # Label is the folder name
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
    
    # Logging statistics
    print(f"Processed {len(features)} images in total.")
    for label in set(labels):
        print(f"Number of images for label '{label}': {labels.count(label)}")
    
    return np.array(features), np.array(labels)

