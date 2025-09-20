# omr_pipeline.py
import cv2, numpy as np, json, os, math
from PIL import Image, ImageDraw, ImageFont
import tempfile

def load_template(path):
    with open(path, "r") as f:
        return json.load(f)

def load_answer_key(path):
    with open(path, "r") as f:
        return json.load(f)

# --- helper: order corner points (tl, tr, br, bl) ---
def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Try: detect 4 large square fiducials (registration marks) - Enhanced for better alignment
def detect_registration_marks(img_gray):
    h, w = img_gray.shape
    
    # Method 1: Enhanced corner detection for OMR sheets
    # Look for dark corners or marks near the corners of the image
    corners = []
    margin = min(h, w) // 15  # Reduced margin for better detection
    
    # Define corner regions to search
    corner_regions = [
        (0, 0, margin*2, margin*2),  # Top-left
        (w-margin*2, 0, margin*2, margin*2),  # Top-right
        (w-margin*2, h-margin*2, margin*2, margin*2),  # Bottom-right
        (0, h-margin*2, margin*2, margin*2)  # Bottom-left
    ]
    
    for i, (x, y, rw, rh) in enumerate(corner_regions):
        roi = img_gray[y:y+rh, x:x+rw]
        
        # Find the darkest point in this corner region
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(roi)
        
        # Convert back to full image coordinates
        corner_x = x + min_loc[0]
        corner_y = y + min_loc[1]
        corners.append([corner_x, corner_y])
    
    if len(corners) == 4:
        return np.array(corners, dtype="float32")
    
    # Method 2: Look for circular patterns (timing marks or registration circles)
    circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min(h, w) // 10,
        param1=50,
        param2=25,  # Lower threshold for better detection
        minRadius=min(h, w) // 150,
        maxRadius=min(h, w) // 40
    )
    
    if circles is not None and len(circles[0]) >= 4:
        circles = np.round(circles[0, :]).astype("int")
        corner_marks = []
        margin = min(h, w) // 8
        
        for (x, y, r) in circles:
            near_corner = (
                (x < margin and y < margin) or
                (x > w - margin and y < margin) or  
                (x > w - margin and y > h - margin) or
                (x < margin and y > h - margin)
            )
            if near_corner:
                corner_marks.append([x, y])
        
        if len(corner_marks) >= 4:
            return np.array(corner_marks[:4], dtype="float32")
    
    # Method 3: Enhanced contour-based detection
    # Apply stronger preprocessing for better contour detection
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 500 or area > min(h, w)**2 // 100: continue
        
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        
        if len(approx) >= 4:  # Accept rectangles and approximate squares
            (x, y, wc, hc) = cv2.boundingRect(approx)
            ar = wc / float(hc)
            if 0.5 < ar < 2.0:  # More lenient aspect ratio
                # Check if near corners
                center_x, center_y = x + wc//2, y + hc//2
                near_corner = (
                    (center_x < margin and center_y < margin) or
                    (center_x > w - margin and center_y < margin) or
                    (center_x > w - margin and center_y > h - margin) or
                    (center_x < margin and center_y > h - margin)
                )
                if near_corner:
                    squares.append((area, [center_x, center_y]))
    
    if len(squares) >= 4:
        # Sort by area and take the 4 largest
        squares = sorted(squares, key=lambda x: x[0], reverse=True)[:4]
        centers = [s[1] for s in squares]
        return np.array(centers, dtype="float32")
    
    # If no registration marks found, return None for fallback processing
    return None

def warp_to_template(img, reg_marks=None, target_w=2480, target_h=3508):
    """Enhanced warping with better fallback handling"""
    if reg_marks is not None and len(reg_marks) == 4:
        try:
            pts = order_points(reg_marks)
            dst = np.array([[0,0],[target_w-1,0],[target_w-1,target_h-1],[0,target_h-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(img, M, (target_w, target_h))
            return warped
        except:
            pass  # Fall through to backup method
    
    # Enhanced fallback: Smart cropping and scaling
    h, w = img.shape[:2]
    
    # Try to detect the actual content area (where the OMR bubbles are)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    # Find content boundaries by detecting dark regions (text/bubbles)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours to determine content area
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the bounding box of all content
        all_points = np.vstack([cv2.findNonZero(thresh)])
        x, y, w_content, h_content = cv2.boundingRect(all_points)
        
        # Add some padding
        padding = min(w, h) // 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w_content = min(w - x, w_content + 2*padding)
        h_content = min(h - y, h_content + 2*padding)
        
        # Crop to content area
        cropped = img[y:y+h_content, x:x+w_content]
        
        # Scale to target size while maintaining aspect ratio
        scale_w = target_w / w_content
        scale_h = target_h / h_content
        scale = min(scale_w, scale_h)
        
        new_w = int(w_content * scale)
        new_h = int(h_content * scale)
        
        resized = cv2.resize(cropped, (new_w, new_h))
        
        # Create target-sized image and center the resized content
        warped = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        
        warped[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return warped
    else:
        # Simple resize as last resort
        warped = cv2.resize(img, (target_w, target_h))
        return warped

def classify_bubble(roi_gray, fill_thresh=0.2):
    # roi_gray: grayscale crop
    # Enhanced bubble classification for higher accuracy
    if roi_gray.size == 0:
        return 0.0
    
    # Preprocessing to enhance bubble detection
    # Apply Gaussian blur to smooth out noise
    blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    
    # Apply morphological operations to clean up the ROI
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    roi_cleaned = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    
    # Enhanced edge detection for better bubble boundary detection
    edges = cv2.Canny(roi_cleaned, 50, 150)
    
    # Find contours to detect filled areas
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate filled area using multiple methods for robustness
    methods = []
    
    # Method 1: Adaptive threshold - Mean
    try:
        thr1 = cv2.adaptiveThreshold(roi_cleaned, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 8)
        methods.append(thr1.sum() / 255.0 / (thr1.shape[0] * thr1.shape[1]))
    except:
        pass
    
    # Method 2: Adaptive threshold - Gaussian
    try:
        thr2 = cv2.adaptiveThreshold(roi_cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 6)
        methods.append(thr2.sum() / 255.0 / (thr2.shape[0] * thr2.shape[1]))
    except:
        pass
    
    # Method 3: Otsu's threshold
    try:
        _, thr3 = cv2.threshold(roi_cleaned, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        methods.append(thr3.sum() / 255.0 / (thr3.shape[0] * thr3.shape[1]))
    except:
        pass
    
    # Method 4: Intensity-based detection
    try:
        # Calculate fraction of pixels below a certain intensity threshold
        intensity_threshold = np.mean(roi_cleaned) - np.std(roi_cleaned) * 0.5
        dark_pixels = np.sum(roi_cleaned < intensity_threshold)
        total_pixels = roi_cleaned.shape[0] * roi_cleaned.shape[1]
        methods.append(dark_pixels / total_pixels)
    except:
        pass
    
    if not methods:
        return 0.0
    
    # Use weighted average with preference for more reliable methods
    methods = np.array(methods)
    
    # Remove outliers (values that are too far from the median)
    median_val = np.median(methods)
    mad = np.median(np.abs(methods - median_val))  # Median absolute deviation
    filtered_methods = methods[np.abs(methods - median_val) <= 3 * mad]
    
    if len(filtered_methods) == 0:
        filtered_methods = methods
    
    # Return the mean of filtered methods
    return float(np.mean(filtered_methods))

def process_image(img_path, template_path, answer_key_path, debug=False, fill_thresh=0.2, option_radius=0.02):
    # Load
    img = cv2.imread(img_path)
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Try detect reg marks and warp
    reg = detect_registration_marks(gray)
    warped = warp_to_template(img, reg_marks=reg)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    tpl = load_template(template_path)
    ak = load_answer_key(answer_key_path)
    
    # Check if template uses points format or bubble_grid format
    if "points" in tpl:
        pts_norm = tpl["points"]  # list of normalized (x,y) points in same order you clicked
        # Each set of 4 points corresponds to question option order (A,B,C,D) if you clicked in that order.
        if len(pts_norm) % 4 != 0:
            raise ValueError("Template points should be multiple of 4 (A-D per question). Found: %d" % len(pts_norm))
        qcount = len(pts_norm) // 4
    elif "bubble_grid" in tpl:
        # Generate points from bubble_grid format
        grid = tpl["bubble_grid"]
        
        # Check for ultra-precise layout
        if "layout" in grid and grid["layout"] == "ultra_precise":
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
        
        # Check for accurate manual layout
        elif "layout" in grid and grid["layout"] == "accurate_manual":
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
        
        # Check for visual calibrated layout v2
        elif "layout" in grid and grid["layout"] == "visual_calibrated_v2":
            pts_norm = []
            qcount = 100
            
            start_y = grid["start_y"]
            row_spacing = grid["row_spacing"]
            
            # Process each column with precise coordinates
            for col_info in grid["columns"]:
                start_x = col_info["start_x"]
                option_spacing = col_info["option_spacing"]
                
                # Generate points for this column (20 questions)
                for row in range(20):  # 20 questions per column
                    for opt in range(4):  # A, B, C, D options
                        x = start_x + opt * option_spacing
                        y = start_y + row * row_spacing
                        pts_norm.append([x, y])
        
        # Check for manual precise layout
        elif "layout" in grid and grid["layout"] == "manual_precise":
            pts_norm = []
            qcount = 100
            
            start_y = grid["start_y"]
            row_spacing = grid["row_spacing"]
            
            # Process each column
            for col_info in grid["columns"]:
                start_x = col_info["start_x"]
                option_spacing = col_info["option_spacing"]
                q_start, q_end = col_info["questions"]
                
                # Generate points for this column
                for row in range(20):  # 20 questions per column
                    for opt in range(4):  # A, B, C, D options
                        x = start_x + opt * option_spacing
                        y = start_y + row * row_spacing
                        pts_norm.append([x, y])
        
        # Check if it's the new 5-column layout
        elif "grid_layout" in grid and grid["grid_layout"] == "5_columns_x_20_rows":
            start_x, start_y = grid["start_x"], grid["start_y"]
            column_x_spacing = grid["column_x_spacing"]
            question_y_spacing = grid["question_y_spacing"] 
            option_x_spacing = grid["option_x_spacing"]
            
            # Template dimensions for normalization
            template_width = 2480
            template_height = 3508
            
            pts_norm = []
            qcount = 100
            
            # Layout: 5 columns of questions, 20 rows each, 4 options each (A,B,C,D)
            # Question mapping: Column 0 = Q1-20, Column 1 = Q21-40, etc.
            for col in range(5):  # 5 columns (subjects)
                for row in range(20):  # 20 questions per column
                    for opt in range(4):  # A, B, C, D options
                        x = (start_x + col * column_x_spacing + opt * option_x_spacing) / template_width
                        y = (start_y + row * question_y_spacing) / template_height
                        pts_norm.append([x, y])
        else:
            # Original grid format
            rows, cols = grid["rows"], grid["cols"]
            start_x, start_y = grid["start_x"], grid["start_y"]
            x_spacing, y_spacing = grid["x_spacing"], grid["y_spacing"]
            
            # Assume template dimensions for normalization
            template_width = 2480
            template_height = 3508
            
            pts_norm = []
            for row in range(rows):
                for col in range(cols):
                    x = (start_x + col * x_spacing) / template_width
                    y = (start_y + row * y_spacing) / template_height
                    pts_norm.append([x, y])
            
            qcount = rows  # Each row is a question with 4 options
    else:
        raise ValueError("Template must contain either 'points' or 'bubble_grid' format")

    results = {"qcount": qcount, "questions": {}}

    H, W = warped_gray.shape[:2]
    overlay = warped.copy()
    
    # Create debug directory for alignment verification
    debug_dir = "debug_alignment"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Save the warped image for debugging
    cv2.imwrite(os.path.join(debug_dir, "warped_image.png"), warped)
    
    for q in range(qcount):
        options = {}
        for opt_idx, opt_label in enumerate(["A","B","C","D"]):
            idx = q*4 + opt_idx
            if idx >= len(pts_norm):
                continue
                
            nx, ny = pts_norm[idx]
            cx = int(nx * W)
            cy = int(ny * H)
            
            # Adaptive radius calculation based on image size and expected bubble size
            base_radius = max(15, int(option_radius * min(W,H)))
            
            # Fine-tune radius based on local content
            search_radius = int(base_radius * 1.5)
            x0_search = max(cx-search_radius, 0)
            y0_search = max(cy-search_radius, 0)
            x1_search = min(cx+search_radius, W-1)
            y1_search = min(cy+search_radius, H-1)
            search_roi = warped_gray[y0_search:y1_search, x0_search:x1_search]
            
            if search_roi.size > 0:
                # Find the darkest circle in the search area
                circles = cv2.HoughCircles(
                    search_roi,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=base_radius,
                    param1=50,
                    param2=20,
                    minRadius=base_radius//2,
                    maxRadius=base_radius*2
                )
                
                if circles is not None and len(circles[0]) > 0:
                    # Use the detected circle center
                    best_circle = circles[0][0]
                    cx = int(x0_search + best_circle[0])
                    cy = int(y0_search + best_circle[1])
                    r = max(12, min(int(best_circle[2]), base_radius*2))
                else:
                    r = base_radius
            else:
                r = base_radius
            
            # Extract ROI for classification
            x0,y0 = max(cx-r,0), max(cy-r,0)
            x1,y1 = min(cx+r,W-1), min(cy+r,H-1)
            roi = warped_gray[y0:y1, x0:x1]
            
            if roi.size == 0:
                frac = 0.0
            else:
                frac = classify_bubble(roi)
                
            # Save bubble ROIs for first few questions for debugging
            if q < 5:
                debug_path = os.path.join(debug_dir, f"q{q+1}_{opt_label}_center_{cx}_{cy}_r_{r}_frac{frac:.3f}.png")
                if roi.size > 0:
                    cv2.imwrite(debug_path, roi)
                
            options[opt_label] = {"center": [int(cx),int(cy)], "filled_frac": round(frac,3), "radius": r}
        # determine selected option with enhanced precision logic
        sorted_opts = sorted(options.items(), key=lambda x: x[1]["filled_frac"], reverse=True)
        top_label, top_info = sorted_opts[0]
        second = sorted_opts[1][1]["filled_frac"] if len(sorted_opts) > 1 else 0
        
        # Enhanced selection criteria for better accuracy
        min_fill_threshold = max(0.08, fill_thresh * 0.6)  # Dynamic minimum threshold
        confidence_gap = max(0.04, fill_thresh * 0.25)     # Required gap between top choices
        high_confidence_threshold = fill_thresh * 1.8      # For very confident selections
        
        # Multi-tier decision logic
        if top_info["filled_frac"] < min_fill_threshold:
            # Clearly not filled enough
            selected = None
            note = "blank"
        elif top_info["filled_frac"] >= high_confidence_threshold:
            # Very high confidence - definitely filled
            selected = top_label
            note = "high_confidence"
        elif top_info["filled_frac"] - second < confidence_gap:
            # Check if the top choice is still reasonable
            if top_info["filled_frac"] >= fill_thresh:
                # Marginally acceptable, but mark as ambiguous
                selected = top_label
                note = "marginal"
            else:
                # Too ambiguous to decide
                selected = None
                note = "unclear"
        else:
            # Clear winner with sufficient gap
            selected = top_label
            note = "confident"
        qno = q+1
        correct = ak["answers"].get(str(qno))
        is_correct = (selected == correct)
        results["questions"][str(qno)] = {
            "selected": selected,
            "note": note,
            "options": options,
            "correct": correct,
            "is_correct": bool(is_correct)
        }
        # Draw overlay: circle selected / correct / wrong with adaptive radius
        for lbl,opt in options.items():
            cx,cy = opt["center"]
            r = opt.get("radius", int(option_radius*min(W,H)*0.9))
            color = (200,200,200)
            thickness = 2
            
            if lbl == selected:
                if is_correct:
                    color = (0,255,0)   # green
                    thickness = 3
                elif selected is None:
                    color = (200,200,0) # yellow
                    thickness = 2
                else:
                    color = (0,0,255)   # red  
                    thickness = 3
            elif lbl == correct:
                # Show correct answer in blue if not selected
                color = (255,100,100)  # light blue
                thickness = 1
                
            cv2.circle(overlay, (cx,cy), r, color, thickness)
            
            # Add option label
            font_scale = 0.6
            font_thickness = 1
            text_size = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2
            
            cv2.putText(overlay, lbl, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), font_thickness)

    # compute subject-wise and total scores
    subjects = ak["subjects"]
    subject_scores = []
    total_correct = 0
    for subj in subjects:
        s_from, s_to = subj["from"], subj["to"]
        corrects = 0
        for qn in range(s_from, s_to+1):
            rec = results["questions"].get(str(qn))
            if rec and rec["is_correct"]:
                corrects += 1
        subject_scores.append({"name": subj["name"], "score": int(corrects)})
        total_correct += corrects
    total_score = int(total_correct)
    results["subject_scores"] = subject_scores
    results["total"] = total_score

    # Save overlay image temporarily
    out_img_path = os.path.join(tempfile.gettempdir(), "overlay.png")
    cv2.imwrite(out_img_path, overlay)
    results["overlay_path"] = out_img_path
    return results
