# Verify answer key by manually checking visible filled bubbles in the OMR image
# Based on the provided image, let me create the correct answer key

def analyze_omr_sheet_visual():
    """Analyze the filled bubbles in the provided OMR sheet image"""
    
    print("Manual Analysis of OMR Sheet")
    print("=" * 50)
    
    # Based on visual inspection of the provided image
    # I can see the following filled bubbles (dark circles):
    
    # PYTHON column (Questions 1-20) - Column 1
    python_answers = {
        1: "D",  # D is filled
        2: "D",  # D is filled
        3: "B",  # B is filled
        4: "D",  # D is filled  
        5: "B",  # B is filled
        6: "B",  # B is filled
        7: "A",  # A is filled
        8: "A",  # A is filled
        9: "A",  # A is filled
        10: "A", # A is filled
        11: "C", # C is filled
        12: "D", # D is filled
        13: "D", # D is filled
        14: "A", # A is filled
        15: "A", # A is filled
        16: "A", # A is filled
        17: "C", # C is filled
        18: "D", # D is filled
        19: "D", # D is filled
        20: "B", # B is filled
    }
    
    # DATA ANALYSIS column (Questions 21-40) - Column 2
    data_analysis_answers = {
        21: "A", # A is filled
        22: "D", # D is filled
        23: "B", # B is filled
        24: "A", # A is filled
        25: "D", # D is filled
        26: "A", # A is filled
        27: "A", # A is filled
        28: "A", # A is filled
        29: "C", # C is filled
        30: "D", # D is filled
        31: "D", # D is filled
        32: "A", # A is filled
        33: "C", # C is filled
        34: "C", # C is filled
        35: "A", # A is filled
        36: "A", # A is filled
        37: "C", # C is filled
        38: "B", # B is filled
        39: "A", # A is filled
        40: "B", # B is filled
    }
    
    # MySQL column (Questions 41-60) - Column 3
    mysql_answers = {
        41: "C", # C is filled
        42: "C", # C is filled
        43: "D", # D is filled
        44: "D", # D is filled
        45: "B", # B is filled
        46: "A", # A is filled
        47: "A", # A is filled
        48: "D", # D is filled
        49: "C", # C is filled
        50: "C", # C is filled
        51: "C", # C is filled
        52: "C", # C is filled
        53: "C", # C is filled
        54: "D", # D is filled
        55: "A", # A is filled
        56: "C", # C is filled
        57: "C", # C is filled
        58: "A", # A is filled
        59: "A", # A is filled
        60: "A", # A is filled
    }
    
    # POWER BI column (Questions 61-80) - Column 4
    powerbi_answers = {
        61: "A", # A is filled
        62: "A", # A is filled
        63: "A", # A is filled
        64: "A", # A is filled
        65: "B", # B is filled
        66: "B", # B is filled
        67: "B", # B is filled
        68: "A", # A is filled
        69: "A", # A is filled
        70: "B", # B is filled
        71: "B", # B is filled
        72: "B", # B is filled
        73: "D", # D is filled
        74: "A", # A is filled
        75: "A", # A is filled
        76: "B", # B is filled
        77: "A", # A is filled
        78: "B", # B is filled
        79: "A", # A is filled
        80: "B", # B is filled
    }
    
    # Adv STATS column (Questions 81-100) - Column 5
    stats_answers = {
        81: "A", # A is filled
        82: "A", # A is filled
        83: "A", # A is filled
        84: "A", # A is filled
        85: "C", # C is filled
        86: "C", # C is filled
        87: "B", # B is filled
        88: "B", # B is filled
        89: "D", # D is filled
        90: "B", # B is filled
        91: "A", # A is filled
        92: "A", # A is filled
        93: "D", # D is filled
        94: "D", # D is filled
        95: "B", # B is filled
        96: "D", # D is filled
        97: "A", # A is filled
        98: "A", # A is filled
        99: "B", # B is filled
        100: "D" # D is filled
    }
    
    # Combine all answers
    all_answers = {}
    all_answers.update(python_answers)
    all_answers.update(data_analysis_answers)
    all_answers.update(mysql_answers)
    all_answers.update(powerbi_answers)
    all_answers.update(stats_answers)
    
    # Create the corrected answer key
    corrected_answer_key = {
        "version": "visual_corrected",
        "answers": {str(k): v for k, v in all_answers.items()},
        "subjects": [
            {"name": "PYTHON", "from": 1, "to": 20},
            {"name": "DATA ANALYSIS", "from": 21, "to": 40},
            {"name": "MySQL", "from": 41, "to": 60}, 
            {"name": "POWER BI", "from": 61, "to": 80},
            {"name": "Adv STATS", "from": 81, "to": 100}
        ]
    }
    
    return corrected_answer_key

if __name__ == "__main__":
    corrected_key = analyze_omr_sheet_visual()
    
    # Show some differences with current answer key
    import json
    
    # Load current answer key
    with open("answer_keys/answer_key_precise.json", "r") as f:
        current_key = json.load(f)
    
    print("Comparing with current answer key...")
    differences = 0
    for q in range(1, 101):
        q_str = str(q)
        current_answer = current_key["answers"].get(q_str)
        corrected_answer = corrected_key["answers"].get(q_str)
        
        if current_answer != corrected_answer:
            differences += 1
            if differences <= 10:  # Show first 10 differences
                print(f"Q{q}: Current={current_answer} -> Corrected={corrected_answer}")
    
    print(f"\nTotal differences found: {differences}")
    
    # Save corrected answer key
    with open("answer_keys/answer_key_visual_corrected.json", "w") as f:
        json.dump(corrected_key, f, indent=2)
    
    print("Corrected answer key saved to answer_keys/answer_key_visual_corrected.json")