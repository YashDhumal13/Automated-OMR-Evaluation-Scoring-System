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

# Try: detect 4 large square fiducials (registration marks) - Enhanced for circular patterns
def detect_registration_marks(img_gray):
    h, w = img_gray.shape
    
    # Method 1: Look for circular patterns (like in the provided OMR sheet)
    circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min(h, w) // 8,
        param1=50,
        param2=30,
        minRadius=min(h, w) // 120,
        maxRadius=min(h, w) // 30
    )
    
    if circles is not None and len(circles[0]) >= 4:
        circles = np.round(circles[0, :]).astype("int")
        corner_marks = []
        margin = min(h, w) // 12
        
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
    
    # Method 2: Original square detection (fallback)
    _,th = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1000: continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4:
            (x,y,wc,hc) = cv2.boundingRect(approx)
            ar = wc/float(hc)
            if 0.6 < ar < 1.4:
                squares.append((area, approx.reshape(4,2)))
    if len(squares) < 4:
        return None
    # pick 4 largest
    squares = sorted(squares, key=lambda x: x[0], reverse=True)[:4]
    pts = np.vstack([s[1] for s in squares])
    # reduce to unique 4 by convex hull / kmeans? We will take centroids
    centers = []
    for s in squares:
        M = cv2.moments(s[1])
        if M["m00"] == 0: continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        centers.append([cx,cy])
    if len(centers) != 4:
        return None
    return np.array(centers, dtype="float32")

def warp_to_template(img, reg_marks=None, target_w=2480, target_h=3508):
    if reg_marks is not None:
        pts = order_points(reg_marks)
        dst = np.array([[0,0],[target_w-1,0],[target_w-1,target_h-1],[0,target_h-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(img, M, (target_w, target_h))
        return warped
    else:
        # Fallback: center-crop & resize preserving aspect ratio
        h,w = img.shape[:2]
        scale = target_w / float(w)
        warped = cv2.resize(img, (target_w, int(h*scale)))
        if warped.shape[0] < target_h:
            warped = cv2.copyMakeBorder(warped, 0, target_h-warped.shape[0], 0,0, cv2.BORDER_CONSTANT, value=255)
        warped = warped[:target_h, :target_w]
        return warped

def classify_bubble(roi_gray, fill_thresh=0.2):
    # roi_gray: grayscale crop
    # Use adaptive threshold & fraction of dark pixels
    if roi_gray.size == 0:
        return 0.0
    
    # Apply morphological operations to clean up the ROI
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    roi_cleaned = cv2.morphologyEx(roi_gray, cv2.MORPH_CLOSE, kernel)
    
    # Try multiple threshold methods for better detection
    methods = []
    
    try:
        thr1 = cv2.adaptiveThreshold(roi_cleaned, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8)
        methods.append(thr1.sum() / 255.0 / (thr1.shape[0] * thr1.shape[1]))
    except:
        pass
    
    try:
        thr2 = cv2.adaptiveThreshold(roi_cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)
        methods.append(thr2.sum() / 255.0 / (thr2.shape[0] * thr2.shape[1]))
    except:
        pass
    
    try:
        _, thr3 = cv2.threshold(roi_cleaned, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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
        
        # Check for visual calibrated layout
        elif "layout" in grid and grid["layout"] == "visual_calibrated":
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
    
    # Create debug directory
    debug_dir = "debug_bubbles"
    os.makedirs(debug_dir, exist_ok=True)
    
    for q in range(qcount):
        options = {}
        for opt_idx, opt_label in enumerate(["A","B","C","D"]):
            idx = q*4 + opt_idx
            if idx >= len(pts_norm):
                continue
                
            nx, ny = pts_norm[idx]
            cx = int(nx * W)
            cy = int(ny * H)
            r = max(12, int(option_radius * min(W,H)))  # Smaller minimum radius
            
            x0,y0 = max(cx-r,0), max(cy-r,0)
            x1,y1 = min(cx+r,W-1), min(cy+r,H-1)
            roi = warped_gray[y0:y1, x0:x1]
            
            if roi.size == 0:
                frac = 0.0
            else:
                frac = classify_bubble(roi)
                
            # Save first few bubble ROIs for debugging
            if q < 10:
                debug_path = os.path.join(debug_dir, f"q{q+1}_{opt_label}_frac{frac:.3f}.png")
                cv2.imwrite(debug_path, roi)
                
            options[opt_label] = {"center": [int(cx),int(cy)], "filled_frac": round(frac,3)}
        # determine selected option with ultra-precise logic
        sorted_opts = sorted(options.items(), key=lambda x: x[1]["filled_frac"], reverse=True)
        top_label, top_info = sorted_opts[0]
        second = sorted_opts[1][1]["filled_frac"] if len(sorted_opts) > 1 else 0
        
        # Ultra-precise selection criteria
        min_fill_threshold = fill_thresh * 0.75  # Dynamic threshold
        confidence_gap = max(0.05, fill_thresh * 0.3)  # Dynamic confidence gap
        
        # Additional validation: check if top choice is significantly dark
        if top_info["filled_frac"] < min_fill_threshold:
            selected = None
            note = "blank"
        elif top_info["filled_frac"] - second < confidence_gap:
            # For ambiguous cases, apply additional criteria
            if top_info["filled_frac"] > fill_thresh * 1.5:  # Still reasonably filled
                selected = top_label
                note = "ambiguous"
            else:
                selected = None
                note = "unclear"
        else:
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
        # Draw overlay: circle selected / correct / wrong
        for lbl,opt in options.items():
            cx,cy = opt["center"]
            color = (200,200,200)
            if lbl == selected:
                if is_correct:
                    color = (0,200,0)   # green
                elif selected is None:
                    color = (200,200,0)
                else:
                    color = (0,0,200)   # red
            cv2.circle(overlay, (cx,cy), int(option_radius*min(W,H)*0.9), color, 2)

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
