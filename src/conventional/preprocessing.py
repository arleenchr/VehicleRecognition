from skimage.feature import hog
import cv2
import numpy as np

def preprocess_image(pil_image):
    # Convert PIL image to numpy array
    img = np.array(pil_image)
    
    # Convert RGB to BGR for OpenCV processing
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to a fixed size
    resized = cv2.resize(gray, (128, 128))
    
    # Normalize pixel values to [0, 1]
    normalized = resized / 255.0
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny((blurred * 255).astype(np.uint8), 100, 200)
    
    # Extract HOG features
    hog_features, hog_image = hog(
        normalized, 
        orientations=9, 
        pixels_per_cell=(8, 8), 
        cells_per_block=(2, 2), 
        block_norm='L2-Hys', 
        visualize=True, 
        transform_sqrt=True
    )
    
    # Enhance HOG visualization by scaling to 0-255 and converting to uint8
    hog_image = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min())  # Normalize to 0-1
    hog_image = (hog_image * 255).astype(np.uint8)  # Scale to 0-255
        
    combined_features = np.concatenate([hog_features, edges.flatten()])
    
    return combined_features, hog_image