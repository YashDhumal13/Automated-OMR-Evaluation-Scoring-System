# app.py
import streamlit as st, os, json, time, sqlite3, pandas as pd
from omr_pipeline import process_image
from pathlib import Path
import tempfile
from PIL import Image
import traceback

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

st.set_page_config(layout="wide", page_title="OMR Auto-Eval — MVP")

# Display current configuration first
st.sidebar.header("System Info")
st.sidebar.info(f"Python Path: {SCRIPT_DIR}")
st.sidebar.info(f"Upload size limit: 200MB")

st.sidebar.header("Config")
template_path = st.sidebar.text_input("Template JSON", str(SCRIPT_DIR / "templates" / "template_visual_calibrated_v2.json"))
answer_key_path = st.sidebar.text_input("Answer key JSON", str(SCRIPT_DIR / "answer_keys" / "answer_key_visual_corrected.json"))
fill_thresh = st.sidebar.slider("Fill threshold", 0.05, 0.6, 0.12, 0.01)  # Lower threshold for better accuracy
option_radius = st.sidebar.slider("Option radius (norm)", 0.005, 0.05, 0.020, 0.001)  # Slightly larger radius for better detection

st.title("Automated OMR Evaluation & Scoring — MVP")

# Instructions for fixing upload issues
with st.expander("📋 Instructions & Troubleshooting", expanded=False):
    st.markdown("""
    ### How to Process Your OMR Sheet:
    
    **Option 1: File Upload (Recommended)**
    - Click "Browse files" below and select your OMR image
    - Supported formats: JPG, JPEG, PNG
    - Max file size: 200MB
    
    **Option 2: Direct File Path (If upload fails)**
    - Save your OMR image to a local folder
    - Enter the full path in the text box (e.g., `C:\\Users\\YourName\\Desktop\\omr_sheet.jpg`)
    - Click Process
    
    **Option 3: Demo Test**
    - Use the demo checkbox to test with a sample image
    
    ### If you see "AxiosError: Request failed with status code 403":
    - Try refreshing the page (Ctrl+F5)
    - Use the direct file path option instead
    - Make sure your file is not too large (>200MB)
    """)

# Current configuration display
col1, col2 = st.columns(2)
with col1:
    st.info(f"📁 Current directory: `{SCRIPT_DIR}`")
with col2:
    st.info(f"🔧 Template: `{os.path.basename(template_path)}`")

uploaded = st.file_uploader("Upload OMR image(s)", type=["jpg","jpeg","png"], accept_multiple_files=True, help="Select your OMR sheet images to process")

# Add alternative options for upload issues
st.write("**Alternative Options:**")
col1, col2 = st.columns(2)

with col1:
    use_test_image = st.checkbox("Use demo test image")

with col2:
    direct_path = st.text_input("Or enter direct file path:", placeholder="C:\\path\\to\\your\\image.jpg")

if st.button("Process") and (uploaded or use_test_image or direct_path):
    results_dir = SCRIPT_DIR / "results_out"
    os.makedirs(results_dir, exist_ok=True)
    conn = sqlite3.connect(str(SCRIPT_DIR / "results.db"))
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY, filename TEXT, version TEXT, total INTEGER, result_json TEXT, ts REAL)""")
    
    # Handle different input methods
    if direct_path and os.path.exists(direct_path):
        files_to_process = [{"name": os.path.basename(direct_path), "path": direct_path}]
        st.success(f"Using direct path: {direct_path}")
    elif use_test_image and not uploaded:
        # Create a test file list with a sample image if available
        test_files = []
        test_image_path = SCRIPT_DIR / "test_sample.jpg" 
        if test_image_path.exists():
            test_files.append({"name": "test_sample.jpg", "path": str(test_image_path)})
            st.success("Using demo test image")
        else:
            st.error("No test image available. Please upload an image instead.")
            st.stop()
        files_to_process = test_files
    elif uploaded:
        files_to_process = uploaded
        st.success(f"Processing {len(uploaded)} uploaded file(s)")
    else:
        st.error("Please select a file to process.")
        st.stop()
    
    for file in files_to_process:
        if isinstance(file, dict):  # Direct path or test image
            fn = file["name"]
            tmp = file["path"]
        else:  # Uploaded file
            fn = file.name
            tmp = os.path.join(tempfile.gettempdir(), fn)
            try:
                with open(tmp, "wb") as f:
                    f.write(file.getbuffer())
            except Exception as e:
                st.error(f"Error saving uploaded file {fn}: {str(e)}")
                continue
                
        st.info("Processing " + fn)
        
        # Verify file exists and is readable
        if not os.path.exists(tmp):
            st.error(f"File not found: {tmp}")
            continue
            
        try:
            res = process_image(tmp, template_path, answer_key_path, fill_thresh=fill_thresh, option_radius=option_radius)
            st.image(res["overlay_path"], caption=f"Overlay: {fn}")
            st.write("Total score:", res["total"])
            for s in res["subject_scores"]:
                st.write(s["name"], ":", s["score"])
            # save in DB
            c.execute("INSERT INTO results (filename, version, total, result_json, ts) VALUES (?,?,?,?,?)",
                      (fn, Path(template_path).stem, res["total"], json.dumps(res), time.time()))
            conn.commit()
            st.success("Saved result for " + fn)
        except Exception as e:
            st.error(f"Error processing {fn}: {str(e)}")
            # Show detailed error information
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            continue
    conn.close()
    st.success("Batch done.")

if st.button("Export CSV"):
    try:
        conn = sqlite3.connect(str(SCRIPT_DIR / "results.db"))
        # Check if table exists
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'")
        if c.fetchone():
            df = pd.read_sql_query("SELECT id, filename, version, total, ts FROM results", conn)
            if not df.empty:
                ts = int(time.time())
                out_csv = str(SCRIPT_DIR / f"results_{ts}.csv")
                df.to_csv(out_csv, index=False)
                conn.close()
                with open(out_csv, "rb") as f:
                    st.download_button("Download CSV", f, file_name=out_csv)
                st.success(f"CSV exported: {out_csv}")
            else:
                st.warning("No results to export yet. Process some OMR images first.")
        else:
            st.warning("No results to export yet. Process some OMR images first.")
        conn.close()
    except Exception as e:
        st.error(f"Error exporting CSV: {str(e)}")
