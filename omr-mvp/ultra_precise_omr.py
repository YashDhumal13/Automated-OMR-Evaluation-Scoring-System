# ultra_precise_omr.py - Maximum accuracy OMR system
import cv2
import numpy as np
import json
import os
import tempfile
from pathlib import Path

def detect_enhanced_registration_marks(img_gray):
    """Enhanced registration mark detection using the corner circles"""
    h, w = img_gray.shape
    
    # Look for circular patterns in corners
    # Use HoughCircles to detect the circular registration marks
    circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min(h, w) // 8,  # Minimum distance between circles
        param1=50,
        param2=30,
        minRadius=min(h, w) // 100,  # Minimum circle radius
        maxRadius=min(h, w) // 20   # Maximum circle radius  
    )
    
    if circles is not None and len(circles[0]) >= 4:
        circles = np.round(circles[0, :]).astype("int")
        
        # Find corner marks (should be near edges)
        corner_marks = []
        margin = min(h, w) // 10
        
        for (x, y, r) in circles:
            is_corner = (
                (x < margin or x > w - margin) and
                (y < margin or y > h - margin)
            )
            if is_corner:
                corner_marks.append([x, y])
        
        if len(corner_marks) >= 4:
            return np.array(corner_marks[:4], dtype="float32")
    
    # Fallback: detect dark squares in corners
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    squares = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 500 < area < 5000:  # Reasonable area for registration marks
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) >= 4:  # Roughly rectangular
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    squares.append([cx, cy])
    
    if len(squares) >= 4:
        return np.array(squares[:4], dtype="float32")
    
    return None

def enhanced_perspective_correction(img, reg_marks=None, target_w=2480, target_h=3508):
    """Enhanced perspective correction with better registration mark handling"""
    
    if reg_marks is not None and len(reg_marks) >= 4:
        # Order points: top-left, top-right, bottom-right, bottom-left
        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # top-left
            rect[2] = pts[np.argmax(s)]  # bottom-right
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # top-right
            rect[3] = pts[np.argmax(diff)]  # bottom-left
            return rect
        
        src_pts = order_points(reg_marks)
        dst_pts = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]
        ], dtype="float32")
        
        # Get perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (target_w, target_h))
        return warped
    else:
        # Fallback: smart crop and resize
        h, w = img.shape[:2]
        
        # Calculate aspect ratio
        target_ratio = target_w / target_h
        current_ratio = w / h
        
        if current_ratio > target_ratio:
            # Image is wider, crop sides
            new_w = int(h * target_ratio)
            start_x = (w - new_w) // 2
            cropped = img[:, start_x:start_x + new_w]
        else:
            # Image is taller, crop top/bottom
            new_h = int(w / target_ratio)
            start_y = (h - new_h) // 2
            cropped = img[start_y:start_y + new_h, :]
        
        # Resize to target dimensions
        return cv2.resize(cropped, (target_w, target_h))

def ultra_precise_bubble_classification(roi_gray):
    """Ultra-precise bubble classification with multiple validation methods"""
    if roi_gray.size == 0:
        return 0.0
    
    # Step 1: Image preprocessing
    # Apply Gaussian blur to reduce noise
    roi_blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    roi_cleaned = cv2.morphologyEx(roi_blurred, cv2.MORPH_CLOSE, kernel)
    
    # Step 2: Multiple threshold methods
    fill_fractions = []
    
    # Method 1: Adaptive threshold (mean)
    try:
        thr_adaptive_mean = cv2.adaptiveThreshold(
            roi_cleaned, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, 11, 5
        )
        fill_fractions.append(thr_adaptive_mean.sum() / 255.0 / thr_adaptive_mean.size)
    except:
        pass
    
    # Method 2: Adaptive threshold (Gaussian)
    try:
        thr_adaptive_gauss = cv2.adaptiveThreshold(
            roi_cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 5
        )
        fill_fractions.append(thr_adaptive_gauss.sum() / 255.0 / thr_adaptive_gauss.size)
    except:
        pass
    
    # Method 3: OTSU threshold
    try:
        _, thr_otsu = cv2.threshold(roi_cleaned, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        fill_fractions.append(thr_otsu.sum() / 255.0 / thr_otsu.size)
    except:
        pass
    
    # Method 4: Fixed threshold based on image statistics
    try:
        mean_val = np.mean(roi_cleaned)
        std_val = np.std(roi_cleaned)
        thresh_val = max(0, min(255, mean_val - 0.5 * std_val))
        _, thr_fixed = cv2.threshold(roi_cleaned, thresh_val, 255, cv2.THRESH_BINARY_INV)
        fill_fractions.append(thr_fixed.sum() / 255.0 / thr_fixed.size)
    except:
        pass
    
    if not fill_fractions:
        return 0.0
    
    # Step 3: Statistical analysis of results
    fill_fractions = np.array(fill_fractions)
    
    # Remove extreme outliers (beyond 2 standard deviations)
    if len(fill_fractions) > 2:
        mean_frac = np.mean(fill_fractions)
        std_frac = np.std(fill_fractions)
        mask = np.abs(fill_fractions - mean_frac) <= 2 * std_frac
        fill_fractions = fill_fractions[mask]
    
    # Return median for robustness
    return float(np.median(fill_fractions))

def create_ultra_precise_template():
    """Create ultra-precise template based on detailed image analysis"""
    
    # Based on the provided OMR sheet image analysis
    # The sheet has 5 columns with 20 rows each
    # Blue guide circles at bottom show expected positions
    
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
            "layout": "ultra_precise",
            "description": "Ultra-precise coordinates based on detailed image analysis",
            "image_width": 2480,
            "image_height": 3508,
            "columns": [
                {
                    "name": "PYTHON",
                    "questions": [1, 20],
                    "start_x": 0.074,  # Fine-tuned based on image
                    "option_spacing": 0.0315  # Precise spacing
                },
                {
                    "name": "DATA ANALYSIS",
                    "questions": [21, 40], 
                    "start_x": 0.266,  # Adjusted
                    "option_spacing": 0.0315
                },
                {
                    "name": "MySQL",
                    "questions": [41, 60],
                    "start_x": 0.458,  # Center column
                    "option_spacing": 0.0315
                },
                {
                    "name": "POWER BI",
                    "questions": [61, 80],
                    "start_x": 0.650,  # Fourth column
                    "option_spacing": 0.0315
                },
                {
                    "name": "Adv STATS",
                    "questions": [81, 100],
                    "start_x": 0.842,  # Rightmost
                    "option_spacing": 0.0315
                }
            ],
            "start_y": 0.268,  # Top of question area
            "row_spacing": 0.0308  # Fine-tuned row spacing
        }
    }
    
    return template

def process_image_ultra_precise(img_path, answer_key_path, debug=True):
    """Ultra-precise OMR processing with maximum accuracy"""
    
    print(f"Processing image: {img_path}")
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhanced image preprocessing
    # Apply slight Gaussian blur to reduce noise
    gray_smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Detect registration marks with enhanced method
    print("Detecting registration marks...")
    reg_marks = detect_enhanced_registration_marks(gray_smooth)
    
    # Apply perspective correction
    print("Applying perspective correction...")
    warped = enhanced_perspective_correction(img, reg_marks)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Load template and answer key
    template = create_ultra_precise_template()
    with open(answer_key_path, 'r') as f:
        answer_key = json.load(f)
    
    # Generate bubble coordinates
    grid = template["bubble_grid"]
    start_y = grid["start_y"]
    row_spacing = grid["row_spacing"]
    
    bubble_coords = []
    for col_info in grid["columns"]:
        start_x = col_info["start_x"]
        option_spacing = col_info["option_spacing"]
        
        for row in range(20):  # 20 questions per column
            for opt in range(4):  # A, B, C, D
                x = start_x + opt * option_spacing
                y = start_y + row * row_spacing
                bubble_coords.append([x, y])
    
    # Process bubbles
    H, W = warped_gray.shape[:2]
    results = {"questions": {}, "total": 0, "subject_scores": []}
    overlay = warped.copy()
    
    # Create enhanced debug directory
    if debug:
        debug_dir = "debug_ultra_precise"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save warped image for inspection
        cv2.imwrite(os.path.join(debug_dir, "warped_image.jpg"), warped)
    
    print("Processing bubbles...")
    for q in range(100):
        question_num = q + 1
        options = {}
        
        for opt_idx, opt_label in enumerate(["A", "B", "C", "D"]):
            coord_idx = q * 4 + opt_idx
            if coord_idx >= len(bubble_coords):
                continue
            
            x_norm, y_norm = bubble_coords[coord_idx]
            cx = int(x_norm * W)
            cy = int(y_norm * H)
            
            # Optimal radius based on image resolution
            radius = max(10, int(0.015 * min(W, H)))
            
            # Extract ROI
            x0, y0 = max(cx - radius, 0), max(cy - radius, 0)
            x1, y1 = min(cx + radius, W - 1), min(cy + radius, H - 1)
            roi = warped_gray[y0:y1, x0:x1]
            
            if roi.size == 0:
                fill_fraction = 0.0
            else:
                fill_fraction = ultra_precise_bubble_classification(roi)
            
            options[opt_label] = {
                "center": [cx, cy],
                "filled_frac": round(fill_fraction, 4)
            }
            
            # Save debug images for first 15 questions
            if debug and q < 15:
                debug_path = os.path.join(debug_dir, f"q{question_num}_{opt_label}_frac{fill_fraction:.4f}.png")
                if roi.size > 0:
                    cv2.imwrite(debug_path, roi)
        
        # Enhanced selection logic
        sorted_options = sorted(options.items(), key=lambda x: x[1]["filled_frac"], reverse=True)
        
        if len(sorted_options) < 2:
            selected = None
            confidence = "low"
        else:
            top_option, top_data = sorted_options[0]
            second_data = sorted_options[1][1]
            
            top_frac = top_data["filled_frac"]
            second_frac = second_data["filled_frac"]
            
            # Enhanced thresholds based on analysis
            min_fill_threshold = 0.15
            confidence_gap = 0.06
            
            if top_frac < min_fill_threshold:
                selected = None
                confidence = "blank"
            elif top_frac - second_frac < confidence_gap:
                selected = top_option
                confidence = "ambiguous"
            else:
                selected = top_option
                confidence = "high"
        
        # Check correctness
        correct_answer = answer_key["answers"].get(str(question_num), "?")
        is_correct = (selected == correct_answer)
        
        results["questions"][str(question_num)] = {
            "selected": selected,
            "correct": correct_answer,
            "is_correct": is_correct,
            "confidence": confidence,
            "options": options
        }
        
        # Enhanced visualization
        for opt_label, opt_data in options.items():
            cx, cy = opt_data["center"]
            fill_frac = opt_data["filled_frac"]
            
            if opt_label == selected:
                if is_correct:
                    color = (0, 255, 0)  # Green for correct
                    thickness = 3
                elif selected is None:
                    color = (255, 255, 0)  # Yellow for blank
                    thickness = 2
                else:
                    color = (0, 0, 255)  # Red for incorrect
                    thickness = 3
            else:
                # Color code based on fill fraction for debugging
                intensity = min(255, int(fill_frac * 500))
                color = (intensity, intensity, intensity)
                thickness = 1
            
            cv2.circle(overlay, (cx, cy), radius - 2, color, thickness)
    
    # Calculate scores
    total_correct = 0
    subject_scores = []
    
    for subject in answer_key["subjects"]:
        subject_name = subject["name"]
        start_q = subject["from"]
        end_q = subject["to"]
        
        subject_correct = 0
        for q_num in range(start_q, end_q + 1):
            if results["questions"].get(str(q_num), {}).get("is_correct", False):
                subject_correct += 1
        
        subject_scores.append({
            "name": subject_name,
            "score": subject_correct,
            "total": end_q - start_q + 1
        })
        total_correct += subject_correct
    
    results["total"] = total_correct
    results["subject_scores"] = subject_scores
    
    # Save overlay
    overlay_path = os.path.join(tempfile.gettempdir(), "ultra_precise_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    results["overlay_path"] = overlay_path
    
    print(f"Processing complete! Total score: {total_correct}/100")
    return results

def save_ultra_precise_template():
    """Save the ultra-precise template"""
    template = create_ultra_precise_template()
    
    os.makedirs("templates", exist_ok=True)
    output_path = "templates/template_ultra_precise.json"
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"✅ Ultra-precise template saved: {output_path}")
    return output_path

if __name__ == "__main__":
    save_ultra_precise_template()
    print("Ultra-precise OMR system ready!")
    print("Usage: process_image_ultra_precise('image.jpg', 'answer_key.json')")