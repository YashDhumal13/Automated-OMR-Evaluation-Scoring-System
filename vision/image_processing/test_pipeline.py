"""
Test Script for OMR Image Processing Engine

This script tests the complete OMR processing pipeline using the provided sample data.
It demonstrates how to use the system and validates that all components work together.
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Add the image_processing module to the path
sys.path.append(str(Path(__file__).parent))

# Import modules directly to avoid relative import issues
from pipeline import OMRProcessor, process_omr_image
from modules.scoring import ScoringConfig


def test_single_image():
    """Test processing a single image."""
    print("="*60)
    print("TESTING SINGLE IMAGE PROCESSING")
    print("="*60)
    
    # Use the first image from Set A as a test
    image_path = r"d:\Automated-OMR-Evaluation-Scoring-System\vision\sample data\Set A\Img1.jpeg"
    answer_key_path = r"d:\Automated-OMR-Evaluation-Scoring-System\vision\sample data\Key (Set A and B).xlsx"
    
    print(f"Testing with image: {image_path}")
    print(f"Answer key: {answer_key_path}")
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return False
        
    if not os.path.exists(answer_key_path):
        print(f"⚠️  Warning: Answer key file not found: {answer_key_path}")
        print("Will proceed without answer key (no scoring)")
        answer_key_path = None
    
    try:
        # Process the image
        results = process_omr_image(
            image_path=image_path,
            answer_key_path=answer_key_path,
            student_id="test_student_001",
            debug=True,
            questions_per_row=5,  # Adjust based on your OMR sheet layout
            options_per_question=4  # Assuming A, B, C, D options
        )
        
        # Print results summary
        if results["processing_info"]["success"]:
            print("✅ Processing successful!")
            
            # Image analysis results
            print("\n📊 IMAGE ANALYSIS RESULTS:")
            img_analysis = results["image_analysis"]
            print(f"   Original dimensions: {img_analysis.get('original_dimensions')}")
            print(f"   Corrected dimensions: {img_analysis.get('corrected_dimensions')}")
            
            if "bubble_detection" in img_analysis:
                bubble_info = img_analysis["bubble_detection"]
                print(f"   Questions detected: {bubble_info.get('total_questions')}")
                print(f"   Total bubbles: {bubble_info.get('total_bubbles')}")
                print(f"   Options per question: {bubble_info.get('options_per_question')}")
            
            if "classification_summary" in img_analysis:
                class_summary = img_analysis["classification_summary"]
                print(f"   Answered questions: {class_summary.get('answered_questions')}")
                print(f"   Unanswered questions: {class_summary.get('unanswered_questions')}")
                print(f"   Average confidence: {class_summary.get('average_confidence', 0):.2f}")
            
            # Student answers
            print("\n📝 STUDENT ANSWERS:")
            student_answers = results["student_answers"]
            if student_answers:
                for q_num, answer in sorted(student_answers.items()):
                    status = "✓" if answer else "○"
                    print(f"   Question {q_num}: {answer} {status}")
            else:
                print("   No answers detected")
            
            # Scoring results (if answer key was available)
            if "scoring_results" in results and "summary" in results["scoring_results"]:
                print("\n🎯 SCORING RESULTS:")
                summary = results["scoring_results"]["summary"]
                print(f"   Total Score: {summary.get('total_score')}/{summary.get('total_possible')}")
                print(f"   Percentage: {summary.get('percentage')}%")
                print(f"   Grade: {summary.get('grade')}")
                
                stats = results["scoring_results"]["statistics"]
                print(f"   Correct: {stats.get('correct_answers')}")
                print(f"   Incorrect: {stats.get('incorrect_answers')}")
                print(f"   Unanswered: {stats.get('unanswered_questions')}")
        else:
            print("❌ Processing failed!")
            print(f"Error: {results['processing_info']['error_message']}")
            return False
        
        # Save results
        output_path = "test_results_single.json"
        with open(output_path, 'w') as f:
            import json
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processing():
    """Test batch processing with multiple images."""
    print("\n" + "="*60)
    print("TESTING BATCH PROCESSING")
    print("="*60)
    
    # Get sample images from Set A
    sample_dir = Path(r"d:\Automated-OMR-Evaluation-Scoring-System\vision\sample data\Set A")
    
    if not sample_dir.exists():
        print(f"❌ Error: Sample directory not found: {sample_dir}")
        return False
    
    # Get up to 3 images for testing
    image_files = list(sample_dir.glob("*.jpeg"))[:3]
    
    if not image_files:
        print(f"❌ Error: No image files found in {sample_dir}")
        return False
    
    print(f"Testing with {len(image_files)} images:")
    for img_file in image_files:
        print(f"  - {img_file.name}")
    
    try:
        # Initialize processor
        processor = OMRProcessor()
        
        # Try to load answer key
        answer_key_path = r"d:\Automated-OMR-Evaluation-Scoring-System\vision\sample data\Key (Set A and B).xlsx"
        if os.path.exists(answer_key_path):
            if processor.load_answer_key(answer_key_path):
                print("✅ Answer key loaded successfully")
            else:
                print("⚠️  Warning: Could not load answer key")
        
        # Process batch
        image_paths = [str(img_file) for img_file in image_files]
        student_ids = [f"student_{i+1:03d}" for i in range(len(image_files))]
        
        batch_results = processor.process_batch_images(
            image_paths=image_paths,
            student_ids=student_ids,
            debug=False  # Disable debug for batch to reduce output size
        )
        
        # Print batch summary
        batch_info = batch_results["batch_info"]
        print(f"\n📊 BATCH PROCESSING SUMMARY:")
        print(f"   Total images: {batch_info['total_images']}")
        print(f"   Successful: {batch_info['successful_processing']}")
        print(f"   Failed: {batch_info['failed_processing']}")
        
        # Show individual results summary
        print(f"\n📝 INDIVIDUAL RESULTS:")
        for student_id, result in batch_results["individual_results"].items():
            if result["processing_info"]["success"]:
                answers_count = len([a for a in result.get("student_answers", {}).values() if a])
                score_info = ""
                if "scoring_results" in result and "summary" in result["scoring_results"]:
                    percentage = result["scoring_results"]["summary"]["percentage"]
                    score_info = f" - Score: {percentage:.1f}%"
                print(f"   {student_id}: ✅ {answers_count} answers{score_info}")
            else:
                error = result["processing_info"].get("error_message", "Unknown error")
                print(f"   {student_id}: ❌ Failed - {error}")
        
        # Save batch results
        output_path = "test_results_batch.json"
        processor.save_results_to_file(batch_results, output_path)
        print(f"\n💾 Batch results saved to: {output_path}")
        
        return batch_info['successful_processing'] > 0
        
    except Exception as e:
        print(f"❌ Error during batch testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_system_info():
    """Test system information display."""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    try:
        processor = OMRProcessor()
        system_info = processor.get_system_info()
        
        print(f"Version: {system_info['version']}")
        print(f"\nSupported formats:")
        for format_type, formats in system_info['supported_formats'].items():
            print(f"  {format_type}: {', '.join(formats)}")
        
        print(f"\nComponent configurations:")
        for component, config in system_info['components'].items():
            print(f"  {component}:")
            for param, value in config.items():
                print(f"    {param}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting system info: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("🚀 OMR IMAGE PROCESSING ENGINE - TEST SUITE")
    print("="*60)
    
    # Track test results
    test_results = []
    
    # Test 1: System info
    print("\nTest 1: System Information")
    test_results.append(test_system_info())
    
    # Test 2: Single image processing
    print("\nTest 2: Single Image Processing")
    test_results.append(test_single_image())
    
    # Test 3: Batch processing
    print("\nTest 3: Batch Processing")
    test_results.append(test_batch_processing())
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The OMR processing engine is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    print("\n📖 USAGE INSTRUCTIONS:")
    print("To use the OMR processing engine in your code:")
    print("```python")
    print("from pipeline import process_omr_image")
    print("")
    print("# Simple usage")
    print("results = process_omr_image(")
    print("    image_path='path/to/omr_sheet.jpg',")
    print("    answer_key_path='path/to/answer_key.xlsx',")
    print("    student_id='student_123'")
    print(")")
    print("```")


if __name__ == "__main__":
    main()