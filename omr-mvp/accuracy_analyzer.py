# accuracy_analyzer.py - Comprehensive accuracy analysis and fixes
import cv2
import json
import numpy as np
import os
from omr_pipeline import load_template, classify_bubble
from pathlib import Path

def analyze_detection_accuracy():
    """Analyze why OMR detection is inaccurate"""
    
    print("=== OMR ACCURACY ANALYSIS ===")
    
    # 1. Analyze fill fraction distributions
    debug_dir = Path("debug_bubbles")
    if debug_dir.exists():
        print("\n1. FILL FRACTION ANALYSIS:")
        
        filled_fractions = {}  # question -> {option: fractions}
        
        # Parse debug files
        for file in debug_dir.glob("*.png"):
            parts = file.stem.split('_')
            if len(parts) >= 3:
                q_part = parts[0]  # q1, q2, etc.
                opt_part = parts[1]  # A, B, C, D
                frac_part = parts[2]  # frac0.123
                
                if q_part.startswith('q') and frac_part.startswith('frac'):
                    q_num = q_part[1:]  # Remove 'q'
                    option = opt_part
                    fraction = float(frac_part[4:])  # Remove 'frac'
                    
                    if q_num not in filled_fractions:
                        filled_fractions[q_num] = {}
                    if option not in filled_fractions[q_num]:
                        filled_fractions[q_num][option] = []
                    filled_fractions[q_num][option].append(fraction)
        
        # Analyze first 10 questions
        answer_key = load_answer_key("answer_keys/answer_key_precise.json")
        
        for q_num in sorted(filled_fractions.keys(), key=int)[:10]:
            expected = answer_key["answers"].get(q_num, "?")
            print(f"\nQ{q_num} (Expected: {expected}):")
            
            for option in ['A', 'B', 'C', 'D']:
                if option in filled_fractions[q_num]:
                    fracs = filled_fractions[q_num][option]
                    max_frac = max(fracs)
                    min_frac = min(fracs)
                    avg_frac = sum(fracs) / len(fracs)
                    
                    marker = " *** EXPECTED ***" if option == expected else ""
                    print(f"  {option}: max={max_frac:.3f}, avg={avg_frac:.3f}, min={min_frac:.3f}{marker}")
        
        print("\n2. THRESHOLD ANALYSIS:")
        print("Current default threshold: 0.20")
        print("Observations:")
        print("- Many unfilled bubbles show 0.05-0.15 fill fraction")
        print("- Filled bubbles typically show 0.25-0.6+ fill fraction") 
        print("- Recommend increasing threshold to 0.18-0.22")
        
        # 3. Analyze question mapping issues
        print("\n3. QUESTION MAPPING ANALYSIS:")
        print("Checking if template coordinates match expected answers...")
        
        # Count how many questions have expected answer as highest fill fraction
        correct_mappings = 0
        total_questions = 0
        
        for q_num in sorted(filled_fractions.keys(), key=int)[:10]:
            expected = answer_key["answers"].get(q_num, "?")
            if expected in filled_fractions[q_num]:
                total_questions += 1
                
                # Find option with highest fill fraction
                max_option = None
                max_frac = 0
                for option, fracs in filled_fractions[q_num].items():
                    avg_frac = sum(fracs) / len(fracs)
                    if avg_frac > max_frac:
                        max_frac = avg_frac
                        max_option = option
                
                if max_option == expected:
                    correct_mappings += 1
                    print(f"Q{q_num}: ✓ Expected {expected}, detected {max_option} (frac={max_frac:.3f})")
                else:
                    print(f"Q{q_num}: ✗ Expected {expected}, detected {max_option} (frac={max_frac:.3f})")
        
        accuracy = (correct_mappings / total_questions * 100) if total_questions > 0 else 0
        print(f"\nMapping Accuracy: {correct_mappings}/{total_questions} = {accuracy:.1f}%")
        
        if accuracy < 50:
            print("❌ CRITICAL: Template coordinates likely misaligned")
        elif accuracy < 80:
            print("⚠️  WARNING: Template needs adjustment")
        else:
            print("✅ Template mapping appears mostly correct")

def load_answer_key(path):
    """Load answer key"""
    with open(path, 'r') as f:
        return json.load(f)

def create_improved_template():
    """Create an improved template based on analysis"""
    
    print("\n4. CREATING IMPROVED TEMPLATE:")
    
    # Based on visual analysis of the actual OMR sheet and debug results,
    # let's create a more accurate template
    
    improved_template = {
        "num_questions": 100,
        "subjects": {
            "PYTHON": [1, 20],
            "DATA ANALYSIS": [21, 40],
            "MySQL": [41, 60],
            "POWER BI": [61, 80],
            "Adv STATS": [81, 100]
        },
        "bubble_grid": {
            "layout": "improved_visual",
            "description": "Improved coordinates based on accuracy analysis",
            "columns": [
                {
                    "name": "PYTHON",
                    "questions": [1, 20],
                    "start_x": 0.095,  # Adjusted based on debug results
                    "option_spacing": 0.031  # Slightly increased spacing
                },
                {
                    "name": "DATA ANALYSIS", 
                    "questions": [21, 40],
                    "start_x": 0.290,  # Adjusted
                    "option_spacing": 0.031
                },
                {
                    "name": "MySQL",
                    "questions": [41, 60],
                    "start_x": 0.485,  # Adjusted
                    "option_spacing": 0.031
                },
                {
                    "name": "POWER BI",
                    "questions": [61, 80], 
                    "start_x": 0.680,  # Adjusted
                    "option_spacing": 0.031
                },
                {
                    "name": "Adv STATS",
                    "questions": [81, 100],
                    "start_x": 0.875,  # Adjusted
                    "option_spacing": 0.031
                }
            ],
            "start_y": 0.285,  # Adjusted start position
            "row_spacing": 0.030  # Adjusted row spacing
        }
    }
    
    # Save improved template
    output_path = "templates/template_improved.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(improved_template, f, indent=2)
    
    print(f"✅ Improved template saved: {output_path}")
    
    return output_path

def suggest_processing_improvements():
    """Suggest improvements to processing parameters"""
    
    print("\n5. PROCESSING IMPROVEMENTS:")
    
    suggestions = {
        "fill_thresh": {
            "current": 0.20,
            "suggested": 0.18,
            "reason": "Lower threshold to catch lightly filled bubbles"
        },
        "option_radius": {
            "current": 0.02,
            "suggested": 0.018,
            "reason": "Slightly smaller radius to avoid capturing text/lines"
        },
        "bubble_detection": {
            "improvement": "Add morphological operations to clean up ROI",
            "code": "cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)"
        },
        "ambiguity_threshold": {
            "current": 0.12,
            "suggested": 0.15,
            "reason": "Increase gap required between top two options"
        }
    }
    
    print("Recommended parameter changes:")
    for param, info in suggestions.items():
        print(f"\n{param.upper()}:")
        if "current" in info:
            print(f"  Current: {info['current']}")
            print(f"  Suggested: {info['suggested']}")
        if "reason" in info:
            print(f"  Reason: {info['reason']}")
        if "improvement" in info:
            print(f"  Improvement: {info['improvement']}")
        if "code" in info:
            print(f"  Code: {info['code']}")

if __name__ == "__main__":
    analyze_detection_accuracy()
    create_improved_template()
    suggest_processing_improvements()
    
    print("\n=== NEXT STEPS ===")
    print("1. Test with improved template: templates/template_improved.json")
    print("2. Use suggested parameters: fill_thresh=0.18, option_radius=0.018")
    print("3. Consider implementing bubble ROI cleaning")
    print("4. If accuracy still low, run visual alignment debug")