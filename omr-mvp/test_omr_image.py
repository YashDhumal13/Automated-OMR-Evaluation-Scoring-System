# Test the OMR processing with the uploaded image
import os
from omr_pipeline import process_image
from pathlib import Path
import tempfile
import requests

# Create a test image path
SCRIPT_DIR = Path(__file__).parent.absolute()
test_image_path = str(SCRIPT_DIR / "test_omr_sheet.jpg")

def test_omr_processing():
    """Test OMR processing with current settings"""
    
    template_path = str(SCRIPT_DIR / "templates" / "template_ultra_precise.json")
    answer_key_path = str(SCRIPT_DIR / "answer_keys" / "answer_key_precise.json")
    
    # Test with different configurations
    configurations = [
        {"fill_thresh": 0.15, "option_radius": 0.015, "name": "Ultra-Precise Default"},
        {"fill_thresh": 0.12, "option_radius": 0.018, "name": "High Sensitivity"},
        {"fill_thresh": 0.18, "option_radius": 0.012, "name": "Conservative"},
    ]
    
    for config in configurations:
        print(f"\n=== Testing with {config['name']} ===")
        print(f"Fill threshold: {config['fill_thresh']}")
        print(f"Option radius: {config['option_radius']}")
        
        try:
            # Note: The test_omr_sheet.jpg should be saved first
            if not os.path.exists(test_image_path):
                print("ERROR: Test image not found. Please save the OMR image as 'test_omr_sheet.jpg' first.")
                continue
                
            result = process_image(
                test_image_path, 
                template_path, 
                answer_key_path, 
                fill_thresh=config['fill_thresh'], 
                option_radius=config['option_radius']
            )
            
            print(f"Total Score: {result['total']}/100")
            print("Subject Scores:")
            for subject in result['subject_scores']:
                print(f"  {subject['name']}: {subject['score']}/20")
                
            # Show some specific question results for validation
            print("\nSample Results (First 10 questions):")
            for i in range(1, 11):
                q = result['questions'][str(i)]
                status = "✓" if q['is_correct'] else "✗"
                print(f"  Q{i}: Selected={q['selected']}, Correct={q['correct']} {status}")
                
        except Exception as e:
            print(f"ERROR: {str(e)}")
            
if __name__ == "__main__":
    test_omr_processing()