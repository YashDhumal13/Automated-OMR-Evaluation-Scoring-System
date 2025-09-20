# improved_omr_pipeline.py - Fixed version with better accuracy
import cv2
import numpy as np
import json
import os
import tempfile
from omr_pipeline import load_template, load_answer_key, order_points, detect_registration_marks, warp_to_template

def improved_classify_bubble(roi_gray, fill_thresh=0.18):
    """Improved bubble classification with morphological operations"""
    if roi_gray.size == 0:
        return 0.0
    
    # Apply morphological operations to clean up the ROI
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    roi_cleaned = cv2.morphologyEx(roi_gray, cv2.MORPH_CLOSE, kernel)
    
    # Try multiple threshold methods
    methods = []
    
    # Method 1: Adaptive threshold
    try:
        thr1 = cv2.adaptiveThreshold(roi_cleaned, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 8)
        methods.append(thr1.sum() / 255.0 / (thr1.shape[0] * thr1.shape[1]))
    except:
        pass
    
    # Method 2: OTSU threshold
    try:
        _, thr2 = cv2.threshold(roi_cleaned, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        methods.append(thr2.sum() / 255.0 / (thr2.shape[0] * thr2.shape[1]))
    except:
        pass
    
    # Method 3: Fixed threshold
    try:
        _, thr3 = cv2.threshold(roi_cleaned, 127, 255, cv2.THRESH_BINARY_INV)
        methods.append(thr3.sum() / 255.0 / (thr3.shape[0] * thr3.shape[1]))
    except:
        pass
    
    if not methods:
        return 0.0
    
    # Return the median of the methods to reduce outliers
    methods.sort()
    if len(methods) == 1:
        return float(methods[0])
    elif len(methods) == 2:
        return float((methods[0] + methods[1]) / 2)
    else:
        return float(methods[1])  # median of 3

def create_accurate_template():
    """Create a template with accurate coordinates based on the actual OMR sheet"""
    
    # Based on careful visual analysis of the provided OMR sheet image
    # The layout appears to be 5 columns with 20 rows each
    # Each row has 4 options (A, B, C, D)
    
    template = {
        "num_questions": 100,
        "subjects": {
            "PYTHON": [1, 20],
            "DATA ANALYSIS": [21, 40], 
            "MySQL": [41, 60],
            "POWER BI": [61, 80],
            "Adv STATS": [81, 100]
        },
        "bubble_grid": {
            "layout": "accurate_manual",
            "description": "Manually calibrated coordinates based on actual OMR sheet",
            "columns": [
                {
                    "name": "PYTHON",
                    "questions": [1, 20],
                    "start_x": 0.08,  # Further left adjustment
                    "option_spacing": 0.033  # Proper spacing between A,B,C,D
                },
                {
                    "name": "DATA ANALYSIS", 
                    "questions": [21, 40],
                    "start_x": 0.275,  # Adjusted based on analysis
                    "option_spacing": 0.033
                },
                {
                    "name": "MySQL",
                    "questions": [41, 60],
                    "start_x": 0.47,  # Center column
                    "option_spacing": 0.033
                },
                {
                    "name": "POWER BI",
                    "questions": [61, 80], 
                    "start_x": 0.665,  # Fourth column
                    "option_spacing": 0.033
                },
                {
                    "name": "Adv STATS",
                    "questions": [81, 100],
                    "start_x": 0.86,  # Rightmost column
                    "option_spacing": 0.033
                }
            ],
            "start_y": 0.273,  # Top of first question row
            "row_spacing": 0.0315  # Distance between question rows
        }
    }
    
    return template

def process_image_improved(img_path, template_path, answer_key_path, debug=False, 
                          fill_thresh=0.18, option_radius=0.018):
    """Improved image processing with better accuracy"""
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try detect registration marks and warp
    reg = detect_registration_marks(gray)
    warped = warp_to_template(img, reg_marks=reg)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Load template and answer key
    if template_path.endswith("template_accurate.json"):
        # Use our improved template
        tpl = create_accurate_template()
    else:
        tpl = load_template(template_path)
    
    ak = load_answer_key(answer_key_path)
    
    # Generate template points
    if "bubble_grid" in tpl:
        grid = tpl["bubble_grid"]
        pts_norm = []
        qcount = 100
        
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
        raise ValueError("Template must use bubble_grid format")
    
    results = {"qcount": qcount, "questions": {}}
    H, W = warped_gray.shape[:2]
    overlay = warped.copy()
    
    # Create debug directory
    if debug:
        debug_dir = "debug_bubbles_improved"
        os.makedirs(debug_dir, exist_ok=True)
    
    for q in range(qcount):
        options = {}
        for opt_idx, opt_label in enumerate(["A", "B", "C", "D"]):
            idx = q * 4 + opt_idx
            if idx >= len(pts_norm):
                continue
            
            nx, ny = pts_norm[idx]
            cx = int(nx * W)
            cy = int(ny * H)
            r = max(12, int(option_radius * min(W, H)))  # Smaller radius
            
            # Extract ROI
            x0, y0 = max(cx - r, 0), max(cy - r, 0)
            x1, y1 = min(cx + r, W - 1), min(cy + r, H - 1)
            roi = warped_gray[y0:y1, x0:x1]
            
            if roi.size == 0:
                frac = 0.0
            else:
                frac = improved_classify_bubble(roi, fill_thresh)
            
            # Save debug images for first few questions
            if debug and q < 10:
                debug_path = os.path.join(debug_dir, f"q{q+1}_{opt_label}_frac{frac:.3f}.png")
                cv2.imwrite(debug_path, roi)
            
            options[opt_label] = {"center": [int(cx), int(cy)], "filled_frac": round(frac, 3)}
        
        # Determine selected option with improved logic
        sorted_opts = sorted(options.items(), key=lambda x: x[1]["filled_frac"], reverse=True)
        top_label, top_info = sorted_opts[0]
        second_frac = sorted_opts[1][1]["filled_frac"] if len(sorted_opts) > 1 else 0
        
        # Improved selection logic
        if top_info["filled_frac"] < fill_thresh:
            selected = None
            note = "blank"
        elif top_info["filled_frac"] - second_frac < 0.08:  # Stricter ambiguity check
            selected = top_label
            note = "ambiguous"
        else:
            selected = top_label
            note = "ok"
        
        qno = q + 1
        correct = ak["answers"].get(str(qno))
        is_correct = (selected == correct)
        
        results["questions"][str(qno)] = {
            "selected": selected,
            "note": note,
            "options": options,
            "correct": correct,
            "is_correct": bool(is_correct)
        }
        
        # Draw overlay with better visualization
        for lbl, opt in options.items():
            cx, cy = opt["center"]
            if lbl == selected:
                if is_correct:
                    color = (0, 255, 0)  # green for correct
                    thickness = 3
                elif selected is None:
                    color = (255, 255, 0)  # yellow for blank
                    thickness = 2
                else:
                    color = (0, 0, 255)  # red for wrong
                    thickness = 3
            else:
                color = (128, 128, 128)  # gray for unselected
                thickness = 1
            
            cv2.circle(overlay, (cx, cy), int(option_radius * min(W, H) * 0.8), color, thickness)
    
    # Compute scores
    subjects = ak["subjects"]
    subject_scores = []
    total_correct = 0
    
    for subj in subjects:
        s_from, s_to = subj["from"], subj["to"]
        corrects = 0
        for qn in range(s_from, s_to + 1):
            rec = results["questions"].get(str(qn))
            if rec and rec["is_correct"]:
                corrects += 1
        subject_scores.append({"name": subj["name"], "score": int(corrects)})
        total_correct += corrects
    
    total_score = int(total_correct)
    results["subject_scores"] = subject_scores
    results["total"] = total_score
    
    # Save overlay image
    out_img_path = os.path.join(tempfile.gettempdir(), "overlay_improved.png")
    cv2.imwrite(out_img_path, overlay)
    results["overlay_path"] = out_img_path
    
    return results

def save_accurate_template():
    """Save the accurate template to file"""
    template = create_accurate_template()
    
    os.makedirs("templates", exist_ok=True)
    with open("templates/template_accurate.json", 'w') as f:
        json.dump(template, f, indent=2)
    
    print("✅ Accurate template saved: templates/template_accurate.json")
    return "templates/template_accurate.json"

if __name__ == "__main__":
    save_accurate_template()
    print("Use this template with: process_image_improved()")
    print("Recommended parameters: fill_thresh=0.18, option_radius=0.018")