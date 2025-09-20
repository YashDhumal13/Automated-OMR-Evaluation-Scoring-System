# test_improved_accuracy.py - Test the improved OMR pipeline
import os
import sys
from pathlib import Path

# Add the directory to path so we can import omr_pipeline
sys.path.append(str(Path(__file__).parent))

from omr_pipeline import process_image, load_answer_key

def test_accuracy_improvement():
    """Test the accuracy improvement with the new template and parameters"""
    
    print("=== TESTING IMPROVED OMR ACCURACY ===")
    
    # Test parameters
    template_path = "templates/template_accurate.json" 
    answer_key_path = "answer_keys/answer_key_precise.json"
    fill_thresh = 0.18
    option_radius = 0.018
    
    print(f"Template: {template_path}")
    print(f"Fill threshold: {fill_thresh}")
    print(f"Option radius: {option_radius}")
    
    # Check if template exists
    if not os.path.exists(template_path):
        print(f"❌ Template not found: {template_path}")
        print("Run: python improved_omr_pipeline.py to create it")
        return False
        
    if not os.path.exists(answer_key_path):
        print(f"❌ Answer key not found: {answer_key_path}")
        return False
    
    # Load answer key for comparison
    ak = load_answer_key(answer_key_path)
    
    # Look for test image
    test_image = None
    possible_paths = [
        "test_image.jpg",
        "test_image.png", 
        "sample_omr.jpg",
        "sample_omr.png"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            test_image = path
            break
    
    if not test_image:
        print("ℹ️  No test image found. Place a test OMR image as 'test_image.jpg'")
        print("Available files:")
        for file in os.listdir('.'):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"  - {file}")
        return False
    
    print(f"Test image: {test_image}")
    
    try:
        # Process image with improved settings
        print("\nProcessing image with improved settings...")
        results = process_image(
            test_image, 
            template_path, 
            answer_key_path,
            debug=True,
            fill_thresh=fill_thresh,
            option_radius=option_radius
        )
        
        print(f"\n✅ Processing completed!")
        print(f"Total score: {results['total']}/100")
        
        # Show subject scores
        print("\nSubject scores:")
        for subj in results['subject_scores']:
            print(f"  {subj['name']}: {subj['score']}/20")
        
        # Analyze first 10 questions
        print("\nFirst 10 questions analysis:")
        for q in range(1, 11):
            qdata = results["questions"].get(str(q), {})
            selected = qdata.get("selected", "None")
            correct = qdata.get("correct", "?")
            is_correct = qdata.get("is_correct", False)
            note = qdata.get("note", "")
            
            status = "✓" if is_correct else "✗"
            print(f"  Q{q}: {status} Selected: {selected}, Correct: {correct} ({note})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_accuracy_improvement()