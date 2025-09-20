"""
Simple Demo Script for OMR Processing Engine

This script demonstrates the basic functionality of the OMR processing system
with a simplified approach that avoids complex import issues.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "modules"))

print("🚀 OMR Image Processing Engine - Simple Demo")
print("="*60)

def test_basic_functionality():
    """Test basic image loading and processing capabilities."""
    
    # Test image paths
    sample_dir = Path(r"d:\Automated-OMR-Evaluation-Scoring-System\vision\sample data\Set A")
    
    if not sample_dir.exists():
        print(f"❌ Sample directory not found: {sample_dir}")
        return False
    
    # Get first available image
    image_files = list(sample_dir.glob("*.jpeg"))
    if not image_files:
        print("❌ No JPEG images found in sample directory")
        return False
    
    test_image_path = str(image_files[0])
    print(f"📸 Testing with image: {test_image_path}")
    
    # Test 1: Basic image loading
    print("\n🔍 Test 1: Image Loading")
    try:
        image = cv2.imread(test_image_path)
        if image is None:
            print("❌ Failed to load image")
            return False
        else:
            print(f"✅ Image loaded successfully - Shape: {image.shape}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return False
    
    # Test 2: Perspective Correction Component
    print("\n🔍 Test 2: Perspective Correction")
    try:
        from perspective_correction import PerspectiveCorrector
        
        corrector = PerspectiveCorrector()
        print("✅ PerspectiveCorrector initialized")
        
        # Try basic preprocessing
        preprocessed = corrector.preprocess_image(image)
        print(f"✅ Image preprocessing successful - Shape: {preprocessed.shape}")
        
        # Try edge detection
        edges = corrector.detect_edges(preprocessed)
        print(f"✅ Edge detection successful - Shape: {edges.shape}")
        
    except Exception as e:
        print(f"❌ Error in perspective correction: {e}")
        return False
    
    # Test 3: Bubble Detection Component
    print("\n🔍 Test 3: Bubble Detection")
    try:
        from bubble_detection import BubbleDetector
        
        detector = BubbleDetector()
        print("✅ BubbleDetector initialized")
        
        # Try basic preprocessing for bubbles
        binary_image = detector.preprocess_for_bubbles(image)
        print(f"✅ Bubble preprocessing successful - Shape: {binary_image.shape}")
        
        # Try finding contours
        contours = detector.find_bubble_contours(binary_image)
        print(f"✅ Found {len(contours)} potential bubble contours")
        
    except Exception as e:
        print(f"❌ Error in bubble detection: {e}")
        return False
    
    # Test 4: Scoring Component
    print("\n🔍 Test 4: Scoring System")
    try:
        from scoring import ScoreCalculator, ScoringConfig
        
        calculator = ScoreCalculator()
        print("✅ ScoreCalculator initialized")
        
        # Test with dummy answer key
        dummy_answers = {1: "A", 2: "B", 3: "C", 4: "D", 5: "A"}
        calculator.set_answer_key_manually(dummy_answers)
        print("✅ Answer key set manually")
        
        # Test scoring with dummy student answers
        student_answers = {1: "A", 2: "B", 3: "C", 4: "D", 5: "B"}  # 4/5 correct
        score_result = calculator.calculate_score(student_answers, "test_student")
        print(f"✅ Scoring successful - Score: {score_result.total_score}/{score_result.total_possible}")
        print(f"   Percentage: {score_result.percentage:.1f}%")
        
    except Exception as e:
        print(f"❌ Error in scoring system: {e}")
        return False
    
    return True

def test_opencv_functionality():
    """Test OpenCV specific functionality."""
    print("\n🔍 OpenCV Functionality Test")
    
    try:
        # Test basic OpenCV operations
        test_array = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_array, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        print("✅ Basic OpenCV operations successful")
        print(f"   Original shape: {test_array.shape}")
        print(f"   Grayscale shape: {gray.shape}")
        print(f"   Edges shape: {edges.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenCV functionality error: {e}")
        return False

def test_file_system():
    """Test file system access and structure."""
    print("\n🔍 File System Test")
    
    try:
        # Check sample data structure
        base_path = Path(r"d:\Automated-OMR-Evaluation-Scoring-System\vision")
        sample_data_path = base_path / "sample data"
        
        print(f"Base path exists: {base_path.exists()}")
        print(f"Sample data path exists: {sample_data_path.exists()}")
        
        if sample_data_path.exists():
            set_a = sample_data_path / "Set A"
            set_b = sample_data_path / "Set B"
            answer_key = sample_data_path / "Key (Set A and B).xlsx"
            
            print(f"Set A exists: {set_a.exists()}")
            print(f"Set B exists: {set_b.exists()}")
            print(f"Answer key exists: {answer_key.exists()}")
            
            if set_a.exists():
                images = list(set_a.glob("*.jpeg"))
                print(f"Set A contains {len(images)} images")
                
            if set_b.exists():
                images = list(set_b.glob("*.jpeg"))
                print(f"Set B contains {len(images)} images")
        
        print("✅ File system test completed")
        return True
        
    except Exception as e:
        print(f"❌ File system error: {e}")
        return False

def main():
    """Run the simple demo."""
    
    # Test 1: File system
    fs_success = test_file_system()
    
    # Test 2: OpenCV
    cv_success = test_opencv_functionality()
    
    # Test 3: Component functionality
    component_success = test_basic_functionality()
    
    # Summary
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    
    tests = [
        ("File System", fs_success),
        ("OpenCV Functionality", cv_success), 
        ("Component Functionality", component_success)
    ]
    
    passed = sum(1 for _, success in tests if success)
    total = len(tests)
    
    for test_name, success in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The OMR processing components are working correctly.")
        print("\n📖 USAGE INSTRUCTIONS:")
        print("The image processing engine consists of several components:")
        print("1. PerspectiveCorrector - Transforms skewed images to flat view")
        print("2. BubbleDetector - Finds and organizes answer bubbles") 
        print("3. BubbleClassifier - Determines if bubbles are marked")
        print("4. ScoreCalculator - Compares answers with answer key")
        print("\nEach component can be used independently or combined in a pipeline.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        print("Make sure all required dependencies are installed:")
        print("pip install opencv-python numpy pandas openpyxl")

if __name__ == "__main__":
    main()