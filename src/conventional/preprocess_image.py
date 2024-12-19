from skimage.feature import hog
import cv2
import numpy as np
from constant import IMAGE_HEIGHT, IMAGE_WIDTH

def preprocess_image(pil_image, target_feature_size):
    # Convert PIL image to numpy array
    img_init = np.array(pil_image)

    # Resize the image
    img_resized = cv2.resize(img_init, (IMAGE_WIDTH, IMAGE_HEIGHT))  # Use consistent size

    # Convert to grayscale
    img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    img = cv2.GaussianBlur(img, (9, 9), 0)

    # Extract HOG features and visualization
    hog_features, hog_image = hog(
        img, 
        orientations=9, 
        pixels_per_cell=(8, 8), 
        cells_per_block=(2, 2), 
        block_norm='L2-Hys', 
        transform_sqrt=True,
        visualize=True
    )
    
    # Find the coordinates of the bounding box for HOG features
    binary_mask = (hog_image > 0.9).astype(np.uint8)  # Create a binary mask
    coords = cv2.findNonZero(binary_mask)  # Find non-zero coordinates
    
    # Determine bounding box based on non-zero pixels
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)  # Get bounding box
        top_left = (x, y)
        bottom_right = (x + w, y + h)
    else:
        top_left = (0, 0)
        bottom_right = (IMAGE_WIDTH, IMAGE_HEIGHT)  # Default to full image if no features

    # Bound image using bounding box
    bounded_img = img_resized.copy()
    cv2.rectangle(bounded_img, top_left, bottom_right, 255, 2)

    # Adjust the features to match the target size
    if len(hog_features) > target_feature_size:
        hog_features = hog_features[:target_feature_size]  # Trim excess features
    else:
        hog_features = np.pad(hog_features, (0, target_feature_size - len(hog_features)), mode='constant')  # Pad with zeros
        

    return hog_features, bounded_img