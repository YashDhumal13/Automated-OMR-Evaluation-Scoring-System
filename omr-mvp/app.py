# app.py
import streamlit as st, os, json, time, sqlite3, pandas as pd
from omr_pipeline import process_image
from pathlib import Path
import tempfile
from PIL import Image

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

st.set_page_config(layout="wide", page_title="OMR Auto-Eval — MVP")
st.title("Automated OMR Evaluation & Scoring — MVP")

st.sidebar.header("Config")
template_path = st.sidebar.text_input("Template JSON", str(SCRIPT_DIR / "templates" / "template_ultra_precise.json"))
answer_key_path = st.sidebar.text_input("Answer key JSON", str(SCRIPT_DIR / "answer_keys" / "answer_key_precise.json"))
fill_thresh = st.sidebar.slider("Fill threshold", 0.05, 0.6, 0.15, 0.01)  # Ultra-precise default
option_radius = st.sidebar.slider("Option radius (norm)", 0.005, 0.05, 0.015, 0.001)  # Ultra-precise default

uploaded = st.file_uploader("Upload OMR image(s)", type=["jpg","jpeg","png"], accept_multiple_files=True)
if st.button("Process") and uploaded:
    results_dir = SCRIPT_DIR / "results_out"
    os.makedirs(results_dir, exist_ok=True)
    conn = sqlite3.connect(str(SCRIPT_DIR / "results.db"))
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY, filename TEXT, version TEXT, total INTEGER, result_json TEXT, ts REAL)""")
    for file in uploaded:
        fn = file.name
        tmp = os.path.join(tempfile.gettempdir(), fn)
        with open(tmp, "wb") as f:
            f.write(file.getbuffer())
        st.info("Processing " + fn)
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
            st.error("Error processing " + fn + " : " + str(e))
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
