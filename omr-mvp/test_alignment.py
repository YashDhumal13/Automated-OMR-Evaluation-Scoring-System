# Visual alignment test to verify bubble positioning
import cv2
import numpy as np
from omr_pipeline import process_image
from pathlib import Path
import json

def test_bubble_alignment():
    """Test bubble alignment with the new calibrated template"""
    SCRIPT_DIR = Path.cwd()
    
    # Use the new calibrated template
    template_path = str(SCRIPT_DIR / "templates" / "template_visual_calibrated_v2.json")
    answer_key_path = str(SCRIPT_DIR / "answer_keys" / "answer_key_visual_corrected.json")
    
    print("Testing Bubble Alignment...")
    print("=" * 50)
    print(f"Template: {template_path}")
    print(f"Answer Key: {answer_key_path}")
    
    # Check if files exist
    if not Path(template_path).exists():
        print(f"ERROR: Template not found: {template_path}")
        return
        
    if not Path(answer_key_path).exists():
        print(f"ERROR: Answer key not found: {answer_key_path}")
        return
    
    # Test with sample image (you should save your OMR image as test_omr.jpg)
    test_images = [
        "test_omr.jpg",  # Your actual OMR image
        "test_sample.jpg"  # Demo image
    ]
    
    for img_name in test_images:
        img_path = SCRIPT_DIR / img_name
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            continue
            
        print(f"\nProcessing: {img_name}")
        print("-" * 30)
        
        try:
            # Process with new settings
            result = process_image(
                str(img_path),
                template_path,
                answer_key_path,
                fill_thresh=0.12,
                option_radius=0.020
            )
            
            print(f"✓ Processing successful!")
            print(f"Total Score: {result['total']}/100")
            
            # Show first few question results for verification
            print("\nFirst 5 Questions:")
            for i in range(1, 6):
                if str(i) in result['questions']:
                    q = result['questions'][str(i)]
                    print(f"Q{i}: Selected={q.get('selected', 'None')}, Correct={q.get('correct', 'Unknown')}, Note={q.get('note', 'N/A')}")
            
            print(f"\nOverlay image saved at: {result['overlay_path']}")
            print("Check the debug_alignment folder for detailed bubble analysis.")
            
        except Exception as e:
            print(f"✗ Error processing {img_name}: {str(e)}")
    
    # Display template summary
    print(f"\n" + "="*50)
    print("Template Summary:")
    with open(template_path, 'r') as f:
        template = json.load(f)
    
    grid = template['bubble_grid']
    print(f"Layout: {grid['layout']}")
    print(f"Start Y: {grid['start_y']}")
    print(f"Row Spacing: {grid['row_spacing']}")
    print(f"Columns: {len(grid['columns'])}")
    
    for i, col in enumerate(grid['columns']):
        print(f"  Column {i+1} ({col['name']}): start_x={col['start_x']}, spacing={col['option_spacing']}")

if __name__ == "__main__":
    test_bubble_alignment()