# debug_template.py - Visual debugging tool for bubble positions
import cv2
import json
import numpy as np
from omr_pipeline import load_template

def visualize_bubble_positions(image_path, template_path):
    """Overlay expected bubble positions on the image to see alignment"""
    
    # Load image and template
    img = cv2.imread(image_path)
    tpl = load_template(template_path)
    
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    h, w = img.shape[:2]
    
    # Get bubble positions from template
    if "bubble_grid" in tpl:
        grid = tpl["bubble_grid"]
        
        if "grid_layout" in grid and grid["grid_layout"] == "5_columns_x_20_rows":
            start_x = grid["start_x"]
            start_y = grid["start_y"]
            question_x_spacing = grid["question_x_spacing"]
            question_y_spacing = grid["question_y_spacing"]
            option_x_spacing = grid["option_x_spacing"]
            
            # Template dimensions for normalization
            template_width = 2480
            template_height = 3508
            
            # Draw expected bubble positions
            for row in range(20):  # 20 rows
                for col in range(5):  # 5 columns
                    question_num = row * 5 + col + 1
                    for opt in range(4):  # A, B, C, D options
                        x_norm = (start_x + col * question_x_spacing + opt * option_x_spacing) / template_width
                        y_norm = (start_y + row * question_y_spacing) / template_height
                        
                        # Convert to image coordinates
                        x = int(x_norm * w)
                        y = int(y_norm * h)
                        
                        # Draw circle for expected position
                        cv2.circle(img, (x, y), 15, (0, 255, 0), 2)
                        
                        # Add question number for first option (A)
                        if opt == 0:
                            cv2.putText(img, str(question_num), (x-10, y-20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Save debug image
    debug_path = "debug_bubble_positions.jpg"
    cv2.imwrite(debug_path, img)
    print(f"Debug image saved as: {debug_path}")
    print("Green circles show where algorithm expects bubbles")
    print("Red numbers show question numbers")
    
    return debug_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python debug_template.py <image_path> <template_path>")
        sys.exit(1)
    
    visualize_bubble_positions(sys.argv[1], sys.argv[2])