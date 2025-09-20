# precise_calibration.py - Extract exact bubble coordinates from OMR image
import cv2
import numpy as np
import json

def find_bubble_centers(image_path, output_template_path):
    """
    Automatically detect bubble centers from OMR sheet image
    """
    
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to find dark regions (filled bubbles)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find circles (bubbles) using HoughCircles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=25,  # Minimum distance between circles
        param1=50,   # Higher threshold for edge detection
        param2=30,   # Accumulator threshold for center detection
        minRadius=8, # Minimum circle radius
        maxRadius=20 # Maximum circle radius
    )
    
    bubble_centers = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # Sort circles by position (top to bottom, left to right)
        circles = sorted(circles, key=lambda x: (x[1], x[0]))  # Sort by y, then x
        
        # Create visualization
        debug_img = img.copy()
        for i, (x, y, r) in enumerate(circles):
            cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2)
            cv2.putText(debug_img, str(i), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Normalize coordinates
            h, w = img.shape[:2]
            bubble_centers.append([x/w, y/h])
        
        # Save debug image
        cv2.imwrite("detected_bubbles.jpg", debug_img)
        print(f"Detected {len(circles)} bubbles")
        print("Debug image saved as 'detected_bubbles.jpg'")
    
    # Create template
    template = {
        "num_questions": 100,
        "subjects": {
            "PYTHON": [1, 20],
            "DATA ANALYSIS": [21, 40], 
            "MySQL": [41, 60],
            "POWER BI": [61, 80],
            "Adv STATS": [81, 100]
        },
        "points": bubble_centers
    }
    
    # Save template
    with open(output_template_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Template saved to {output_template_path}")
    return len(bubble_centers)

def manual_grid_template():
    """Create a manually calibrated template based on visual analysis"""
    
    # Based on visual analysis of your OMR sheet
    # Approximate positions for 5 columns x 20 rows x 4 options
    
    template = {
        "num_questions": 100,
        "subjects": {
            "PYTHON": [1, 20],
            "DATA ANALYSIS": [21, 40],
            "MySQL": [41, 60], 
            "POWER BI": [61, 80],
            "Adv STATS": [81, 100]
        },
        "bubble_grid": {
            "layout": "manual_coordinates",
            "coordinates": generate_manual_coordinates()
        }
    }
    
    return template

def generate_manual_coordinates():
    """Generate coordinates based on careful visual analysis"""
    
    # Image dimensions (approximate)
    img_width = 1000
    img_height = 1400
    
    # Starting positions for each column (normalized)
    column_starts = [
        0.12,   # PYTHON column
        0.32,   # DATA ANALYSIS column  
        0.52,   # MySQL column
        0.72,   # POWER BI column
        0.92    # Adv STATS column
    ]
    
    # Y positions for rows (normalized)
    start_y = 0.28
    row_spacing = 0.032
    
    # X spacing between options A, B, C, D
    option_spacing = 0.035
    
    coordinates = []
    
    for col_idx, col_x in enumerate(column_starts):
        for row in range(20):
            for opt in range(4):  # A, B, C, D
                x = col_x + opt * option_spacing
                y = start_y + row * row_spacing
                coordinates.append([x, y])
    
    return coordinates

if __name__ == "__main__":
    # For now, create manual template
    manual_template = manual_grid_template()
    
    with open("templates/template_manual.json", "w") as f:
        json.dump(manual_template, f, indent=2)
    
    print("Manual template created: templates/template_manual.json")