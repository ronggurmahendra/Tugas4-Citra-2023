import cv2
import numpy as np

def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Apply segmentation (contour detection)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through contours
    for contour in contours:
        # Calculate area to filter out small contours
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust the area threshold as needed
            # Draw rectangle around the contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    # Display the result
    cv2.imshow('Detected Vehicles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = './data/2.jpg'
preprocess_image(image_path)