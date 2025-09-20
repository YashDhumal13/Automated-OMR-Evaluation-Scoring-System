# comprehensive_accuracy_test.py - Complete accuracy testing with the ultra-precise system
import os
import sys
import json
import numpy as np
from pathlib import Path

# Add the directory to path
sys.path.append(str(Path(__file__).parent))

from omr_pipeline import process_image, load_answer_key

def comprehensive_accuracy_test():
    """Comprehensive test of the ultra-precise OMR system"""
    
    print("=== ULTRA-PRECISE OMR ACCURACY TEST ===")
    print()
    
    # Configuration
    template_path = "templates/template_ultra_precise.json"
    answer_key_path = "answer_keys/answer_key_precise.json"
    
    # Ultra-precise parameters
    fill_thresh = 0.15  # Lower threshold for better sensitivity
    option_radius = 0.015  # Smaller radius for precision
    
    print(f"🎯 Template: {template_path}")
    print(f"🎯 Fill threshold: {fill_thresh}")  
    print(f"🎯 Option radius: {option_radius}")
    print()
    
    # Validate files exist
    if not os.path.exists(template_path):
        print(f"❌ Template not found: {template_path}")
        print("Run: python ultra_precise_omr.py")
        return False
    
    if not os.path.exists(answer_key_path):
        print(f"❌ Answer key not found: {answer_key_path}")
        return False
    
    # Look for test image
    test_images = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            # Skip debug/output images
            if not any(skip in file.lower() for skip in ['debug', 'overlay', 'warped', 'detected']):
                test_images.append(file)
    
    if not test_images:
        print("❌ No test images found!")
        print("Place OMR image files (.jpg, .png) in the current directory")
        return False
    
    print(f"📁 Found {len(test_images)} test image(s):")
    for img in test_images:
        print(f"  - {img}")
    print()
    
    # Load answer key for analysis
    with open(answer_key_path, 'r') as f:
        answer_key = json.load(f)
    
    # Process each image
    total_tests = 0
    total_accuracy = 0
    
    for img_path in test_images:
        print(f"🔍 Processing: {img_path}")
        print("-" * 50)
        
        try:
            # Process with ultra-precise settings
            results = process_image(
                img_path,
                template_path, 
                answer_key_path,
                debug=True,
                fill_thresh=fill_thresh,
                option_radius=option_radius
            )
            
            # Calculate accuracy metrics
            total_questions = len(results["questions"])
            correct_answers = sum(1 for q in results["questions"].values() if q["is_correct"])
            accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
            
            print(f"✅ Processing complete!")
            print(f"📊 Overall Accuracy: {correct_answers}/{total_questions} = {accuracy:.1f}%")
            print(f"🎯 Total Score: {results['total']}/100")
            print()
            
            # Subject-wise analysis
            print("📚 Subject Scores:")
            for subj in results['subject_scores']:
                subj_accuracy = (subj['score'] / 20 * 100) if 'score' in subj else 0
                print(f"  {subj['name']:<15}: {subj['score']}/20 ({subj_accuracy:.1f}%)")
            print()
            
            # Detailed question analysis (first 20 questions)
            print("🔎 Detailed Analysis (First 20 Questions):")
            confidence_stats = {"confident": 0, "ambiguous": 0, "blank": 0, "unclear": 0}
            
            for q_num in range(1, min(21, total_questions + 1)):
                q_data = results["questions"].get(str(q_num), {})
                selected = q_data.get("selected", "None")
                correct = q_data.get("correct", "?")
                is_correct = q_data.get("is_correct", False)
                note = q_data.get("note", "unknown")
                
                # Get fill fractions for analysis
                options = q_data.get("options", {})
                fractions_str = ", ".join([f"{opt}:{data['filled_frac']:.3f}" 
                                         for opt, data in options.items()])
                
                status = "✅" if is_correct else "❌"
                print(f"  Q{q_num:2d}: {status} Selected: {selected:>4}, Correct: {correct} "
                      f"({note}) [{fractions_str}]")
                
                if note in confidence_stats:
                    confidence_stats[note] += 1
            
            print()
            print("📈 Confidence Statistics:")
            for confidence, count in confidence_stats.items():
                percentage = (count / 20 * 100) if count > 0 else 0
                print(f"  {confidence.capitalize():<10}: {count:2d} ({percentage:4.1f}%)")
            
            total_tests += 1
            total_accuracy += accuracy
            
            # Performance recommendations
            print()
            print("💡 Performance Analysis:")
            if accuracy >= 90:
                print("  🟢 EXCELLENT: System performing at high accuracy")
            elif accuracy >= 80:
                print("  🟡 GOOD: Consider fine-tuning thresholds")
            elif accuracy >= 70:
                print("  🟠 MODERATE: Check template alignment")
            else:
                print("  🔴 LOW: Template or parameters need adjustment")
            
            # Specific recommendations
            ambiguous_rate = confidence_stats.get("ambiguous", 0) / 20 * 100
            blank_rate = confidence_stats.get("blank", 0) / 20 * 100
            
            if ambiguous_rate > 20:
                print(f"  ⚠️  High ambiguity rate ({ambiguous_rate:.1f}%) - consider increasing fill threshold")
            if blank_rate > 10:
                print(f"  ⚠️  High blank rate ({blank_rate:.1f}%) - consider lowering fill threshold")
            
        except Exception as e:
            print(f"❌ Error processing {img_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
        print("\n" + "="*60 + "\n")
    
    # Summary
    if total_tests > 0:
        avg_accuracy = total_accuracy / total_tests
        print(f"📋 FINAL SUMMARY:")
        print(f"   Images processed: {total_tests}")
        print(f"   Average accuracy: {avg_accuracy:.1f}%")
        print()
        
        if avg_accuracy >= 85:
            print("🎉 EXCELLENT RESULTS! Ultra-precise system working well.")
        elif avg_accuracy >= 75:
            print("👍 GOOD RESULTS! Minor fine-tuning may help.")
        else:
            print("🔧 NEEDS IMPROVEMENT! Check template alignment and parameters.")
    
    return True

def analyze_debug_images():
    """Analyze debug images for fine-tuning"""
    
    debug_dir = Path("debug_bubbles")
    if not debug_dir.exists():
        print("No debug images found. Run main test first.")
        return
    
    print("\n🔍 ANALYZING DEBUG IMAGES...")
    
    # Group debug images by question
    questions = {}
    for file in debug_dir.glob("q*_*_frac*.png"):
        parts = file.stem.split('_')
        if len(parts) >= 3:
            q_num = parts[0]
            option = parts[1]  
            frac_part = parts[2]
            
            if q_num not in questions:
                questions[q_num] = {}
            questions[q_num][option] = float(frac_part.replace('frac', ''))
    
    # Analyze fill fraction distributions
    all_fractions = []
    filled_fractions = []
    unfilled_fractions = []
    
    # Load answer key to determine which should be filled
    try:
        with open("answer_keys/answer_key_precise.json", 'r') as f:
            ak = json.load(f)
        
        for q_key, options in questions.items():
            q_num = q_key[1:]  # Remove 'q' prefix
            correct = ak["answers"].get(q_num, "")
            
            for option, fraction in options.items():
                all_fractions.append(fraction)
                if option == correct:
                    filled_fractions.append(fraction)
                else:
                    unfilled_fractions.append(fraction)
        
        if filled_fractions and unfilled_fractions:
            print(f"📊 Fill Fraction Statistics:")
            print(f"   All bubbles: min={min(all_fractions):.3f}, max={max(all_fractions):.3f}, avg={np.mean(all_fractions):.3f}")
            print(f"   Filled (correct): min={min(filled_fractions):.3f}, max={max(filled_fractions):.3f}, avg={np.mean(filled_fractions):.3f}")
            print(f"   Unfilled: min={min(unfilled_fractions):.3f}, max={max(unfilled_fractions):.3f}, avg={np.mean(unfilled_fractions):.3f}")
            
            # Recommend optimal threshold
            filled_avg = np.mean(filled_fractions)
            unfilled_avg = np.mean(unfilled_fractions)
            optimal_thresh = (filled_avg + unfilled_avg) / 2
            print(f"💡 Suggested optimal threshold: {optimal_thresh:.3f}")
    
    except Exception as e:
        print(f"Could not analyze debug images: {e}")

if __name__ == "__main__":
    success = comprehensive_accuracy_test()
    if success:
        analyze_debug_images()
    
    print("\n🎯 NEXT STEPS:")
    print("1. If accuracy is low, adjust template coordinates")
    print("2. Fine-tune fill_thresh based on debug analysis") 
    print("3. Adjust option_radius if bubbles are misaligned")
    print("4. Consider image preprocessing improvements")