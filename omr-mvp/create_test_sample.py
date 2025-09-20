# Create a simple test image for the demo
import cv2
import numpy as np
from pathlib import Path

def create_test_sample():
    """Create a simple test OMR sheet for demonstration"""
    # Create a simple test image (white background with some basic structure)
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255
    
    # Add some text
    cv2.putText(img, "OMR Test Sheet", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "Upload your actual OMR image", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Add some sample circles (not functional, just for demo)
    for i in range(5):
        for j in range(4):
            x = 100 + j * 80
            y = 200 + i * 60
            cv2.circle(img, (x, y), 15, (0, 0, 0), 2)
    
    # Save the test image
    script_dir = Path(__file__).parent
    test_path = script_dir / "test_sample.jpg"
    cv2.imwrite(str(test_path), img)
    print(f"Created test sample at: {test_path}")
    return str(test_path)

if __name__ == "__main__":
    create_test_sample()