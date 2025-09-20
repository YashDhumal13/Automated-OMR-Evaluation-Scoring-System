# OMR MVP - Optical Mark Recognition System

A minimal viable product for processing bubble sheet answer forms using computer vision.

## Structure
- `template_maker.py` - Creates and manages OMR templates
- `omr_pipeline.py` - Main processing pipeline
- `app.py` - GUI application
- `templates/` - Template files
- `answer_keys/` - Answer key files
- `sample_images/` - Test images
- `results.db` - SQLite database for results

## Usage
1. Run `python app.py` to start the GUI
2. Select a template file
3. Process bubble sheet images
4. View results in the interface

## Requirements
- OpenCV
- NumPy
- Tkinter
- SQLite3
