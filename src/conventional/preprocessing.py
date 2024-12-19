from skimage.feature import hog
import cv2
import numpy as np

def preprocess_image(pil_image, target_feature_size, test_image=False):
    # Convert PIL image to numpy array
    img = np.array(pil_image)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize the image
    resized = cv2.resize(gray, (256, 256))  # Use consistent size

    # Normalize pixel values
    normalized = resized / 255.0

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)

    # Extract HOG features and visualization
    hog_features, hog_image = hog(
        blurred, 
        orientations=9, 
        pixels_per_cell=(8, 8), 
        cells_per_block=(2, 2), 
        block_norm='L2-Hys', 
        transform_sqrt=True,
        visualize=True
    )

    # Adjust the features to match the target size
    if len(hog_features) > target_feature_size:
        hog_features = hog_features[:target_feature_size]  # Trim excess features
    else:
        hog_features = np.pad(hog_features, (0, target_feature_size - len(hog_features)), mode='constant')  # Pad with zeros
        
    # Show the HOG image
    if test_image:
        cv2.imshow("HOG Image", hog_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return hog_features, hog_image