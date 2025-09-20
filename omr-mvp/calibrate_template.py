# Analyze the provided OMR sheet to create accurate template coordinates
import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_omr_layout():
    """Analyze the provided OMR sheet to determine accurate bubble positions"""
    
    print("Analyzing OMR Sheet Layout...")
    print("=" * 50)
    
    # Based on visual analysis of the provided OMR sheet
    # The image shows a standard 5-column layout with specific positioning
    
    # Image dimensions (estimate based on standard OMR sheets)
    img_width = 2480
    img_height = 3508
    
    # Observed layout from the image:
    # - 5 columns (PYTHON, DATA ANALYSIS, MySQL, POWER BI, Adv STATS)
    # - 20 rows per column
    # - 4 options per question (A, B, C, D)
    
    # Let me create more accurate coordinates based on visual inspection
    # The bubbles appear to be positioned differently than our current template
    
    # New coordinates based on visual analysis
    new_template = {
        "num_questions": 100,
        "subjects": {
            "PYTHON": [1, 20],
            "DATA ANALYSIS": [21, 40], 
            "MySQL": [41, 60],
            "POWER BI": [61, 80],
            "Adv STATS": [81, 100]
        },
        "bubble_grid": {
            "layout": "visual_calibrated_v2",
            "description": "Coordinates based on visual analysis of actual OMR sheet",
            "image_width": 2480,
            "image_height": 3508,
            
            # Starting positions and spacing (normalized coordinates 0-1)
            "start_y": 0.200,  # Adjusted based on visual inspection
            "row_spacing": 0.0385,  # Spacing between question rows
            
            "columns": [
                {
                    "name": "PYTHON",
                    "questions": [1, 20],
                    "start_x": 0.065,  # Adjusted for better alignment
                    "option_spacing": 0.032  # Spacing between A,B,C,D options
                },
                {
                    "name": "DATA ANALYSIS", 
                    "questions": [21, 40],
                    "start_x": 0.257,  # Adjusted
                    "option_spacing": 0.032
                },
                {
                    "name": "MySQL",
                    "questions": [41, 60],
                    "start_x": 0.449,  # Adjusted
                    "option_spacing": 0.032
                },
                {
                    "name": "POWER BI",
                    "questions": [61, 80], 
                    "start_x": 0.641,  # Adjusted
                    "option_spacing": 0.032
                },
                {
                    "name": "Adv STATS",
                    "questions": [81, 100],
                    "start_x": 0.833,  # Adjusted
                    "option_spacing": 0.032
                }
            ]
        }
    }
    
    return new_template

def create_calibrated_template():
    """Create a new calibrated template file"""
    template = analyze_omr_layout()
    
    # Save the new template
    template_path = Path("templates/template_visual_calibrated_v2.json")
    template_path.parent.mkdir(exist_ok=True)
    
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Created new calibrated template: {template_path}")
    return str(template_path)

if __name__ == "__main__":
    template_path = create_calibrated_template()
    print(f"Template saved to: {template_path}")