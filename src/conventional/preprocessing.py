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
    resized = cv2.resize(gray, (1200, 800))
    
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
    
    # Ensure the edge features are flattened to match the expected size
    edges_flattened = edges.flatten()
    hog_features_length = len(hog_features)
    edges_needed = 2880000 - hog_features_length
    
    # Adjust edge features to match the required size
    if edges_flattened.size > edges_needed:
        edges_flattened = edges_flattened[:edges_needed]
    else:
        edges_flattened = np.pad(edges_flattened, (0, edges_needed - edges_flattened.size), mode='constant')
    
    # Enhance HOG visualization by scaling to 0-255 and converting to uint8
    # hog_image = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min())  # Normalize to 0-1
    # hog_image = (hog_image * 255).astype(np.uint8)  # Scale to 0-255
        
    combined_features = np.concatenate([hog_features, edges_flattened])
    
    return combined_features