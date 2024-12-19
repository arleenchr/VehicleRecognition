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
    
    # Find the coordinates of the bounding box for HOG features
    binary_mask = (hog_image > 0).astype(np.uint8)  # Create a binary mask
    coords = cv2.findNonZero(binary_mask)  # Find non-zero coordinates
    
    # Determine bounding box based on non-zero pixels
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)  # Get bounding box
        top_left = (x, y)
        bottom_right = (x + w, y + h)
    else:
        top_left = (0, 0)
        bottom_right = (hog_image.shape[1], hog_image.shape[0])  # Default to full image if no features

    if test_image:
        # Scale the bounding box coordinates back to the original image dimensions
        scale_x = img.shape[1] / 256  # Original width / resized width
        scale_y = img.shape[0] / 256  # Original height / resized height
        
        original_top_left = (int(top_left[0] * scale_x), int(top_left[1] * scale_y))
        original_bottom_right = (int(bottom_right[0] * scale_x), int(bottom_right[1] * scale_y))
        
        # Convert original image to BGR for color drawing
        original_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Draw the bounding box on the original image
        cv2.rectangle(original_bgr, original_top_left, original_bottom_right, (0, 255, 0), 2)
        
        # Display the original image with the bounding box
        cv2.imshow("Original Image with Bounding Box", original_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Adjust the features to match the target size
    if len(hog_features) > target_feature_size:
        hog_features = hog_features[:target_feature_size]  # Trim excess features
    else:
        hog_features = np.pad(hog_features, (0, target_feature_size - len(hog_features)), mode='constant')  # Pad with zeros
        
    # Show the HOG image
    if test_image:
        # cv2.imshow("HOG Image", hog_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return hog_features, original_bgr
    else:
        return hog_features