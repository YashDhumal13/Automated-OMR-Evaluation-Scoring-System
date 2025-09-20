# visual_debug_alignment.py - Check template alignment with actual image
import cv2
import json
import numpy as np
from omr_pipeline import load_template
import os

def debug_template_alignment(image_path, template_path, answer_key_path):
    """Create a detailed visual debug of template alignment"""
    
    # Load image and template
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
        
    # Load template and answer key
    tpl = load_template(template_path)
    with open(answer_key_path, 'r') as f:
        answer_key = json.load(f)
    
    # Convert to grayscale and warp (same as pipeline)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # For now, assume no warping (you can add registration mark detection later)
    warped = img.copy()
    H, W = warped.shape[:2]
    
    # Generate template points (same logic as pipeline)
    if "bubble_grid" in tpl:
        grid = tpl["bubble_grid"]
        
        if "layout" in grid and grid["layout"] == "visual_calibrated":
            pts_norm = []
            start_y = grid["start_y"]
            row_spacing = grid["row_spacing"]
            
            # Process each column
            for col_info in grid["columns"]:
                start_x = col_info["start_x"]
                option_spacing = col_info["option_spacing"]
                
                # Generate points for this column (20 questions)
                for row in range(20):  # 20 questions per column
                    for opt in range(4):  # A, B, C, D options
                        x = start_x + opt * option_spacing
                        y = start_y + row * row_spacing
                        pts_norm.append([x, y])
        else:
            print("Unsupported template layout")
            return
    else:
        print("Template must have bubble_grid format")
        return
    
    # Create debug visualization
    debug_img = warped.copy()
    
    # Draw expected bubble positions and compare with answer key
    for q_idx in range(100):  # 100 questions
        question_num = q_idx + 1
        
        # Get expected answer from answer key
        expected_answer = answer_key["answers"].get(str(question_num), "?")
        expected_option_idx = ord(expected_answer) - ord('A') if expected_answer in 'ABCD' else -1
        
        # Draw bubbles for this question
        for opt_idx in range(4):  # A, B, C, D
            point_idx = q_idx * 4 + opt_idx
            if point_idx < len(pts_norm):
                x_norm, y_norm = pts_norm[point_idx]
                x = int(x_norm * W)
                y = int(y_norm * H)
                
                # Choose color based on whether this is expected answer
                if opt_idx == expected_option_idx:
                    color = (0, 255, 0)  # Green for expected answer
                    thickness = 3
                else:
                    color = (255, 0, 0)  # Red for other options
                    thickness = 2
                
                # Draw circle
                cv2.circle(debug_img, (x, y), 15, color, thickness)
                
                # Add option letter
                option_letter = chr(ord('A') + opt_idx)
                cv2.putText(debug_img, option_letter, (x-5, y+5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add question number near first option
        if q_idx * 4 < len(pts_norm):
            x_norm, y_norm = pts_norm[q_idx * 4]
            x = int(x_norm * W)
            y = int(y_norm * H)
            cv2.putText(debug_img, f"Q{question_num}({expected_answer})", (x-20, y-25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Save debug image
    debug_path = "template_alignment_debug.jpg"
    cv2.imwrite(debug_path, debug_img)
    print(f"Debug image saved: {debug_path}")
    print("Green circles = Expected answers from answer key")
    print("Red circles = Other options")
    
    return debug_path

if __name__ == "__main__":
    # Test with the provided image
    debug_template_alignment(
        "d:/Code.... Hackathon/omr-mvp/test_image.jpg",  # You'll need to provide actual image path
        "d:/Code.... Hackathon/omr-mvp/templates/template_visual.json",
        "d:/Code.... Hackathon/omr-mvp/answer_keys/answer_key_precise.json"
    )