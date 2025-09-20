"""
OMR MVP Application - Main entry point
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import sqlite3
import json
from omr_pipeline import OMRPipeline
from template_maker import TemplateMaker

class OMRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OMR MVP - Bubble Sheet Reader")
        self.setup_ui()
        self.init_database()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Create main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)
        
        # Template selection
        tk.Label(main_frame, text="Template:").pack()
        self.template_var = tk.StringVar()
        tk.Entry(main_frame, textvariable=self.template_var, width=50).pack()
        tk.Button(main_frame, text="Browse Template", command=self.browse_template).pack()
        
        # Image processing
        tk.Button(main_frame, text="Process Images", command=self.process_images).pack(pady=10)
        
        # Results display
        self.results_text = tk.Text(main_frame, height=10, width=60)
        self.results_text.pack()
    
    def init_database(self):
        """Initialize SQLite database for results"""
        conn = sqlite3.connect('results.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY,
                image_path TEXT,
                answers TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def browse_template(self):
        """Browse for template file"""
        filename = filedialog.askopenfilename(
            title="Select Template",
            filetypes=[("JSON files", "*.json")]
        )
        if filename:
            self.template_var.set(filename)
    
    def process_images(self):
        """Process selected images"""
        template_path = self.template_var.get()
        if not template_path:
            messagebox.showerror("Error", "Please select a template file")
            return
        
        image_paths = filedialog.askopenfilenames(
            title="Select Images to Process",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if image_paths:
            pipeline = OMRPipeline(template_path)
            results = pipeline.process_batch(image_paths)
            self.display_results(results)
            self.save_results(results)
    
    def display_results(self, results):
        """Display processing results"""
        self.results_text.delete(1.0, tk.END)
        for result in results:
            self.results_text.insert(tk.END, f"File: {result['image_path']}\n")
            self.results_text.insert(tk.END, f"Answers: {result['answers']}\n\n")
    
    def save_results(self, results):
        """Save results to database"""
        conn = sqlite3.connect('results.db')
        cursor = conn.cursor()
        for result in results:
            cursor.execute(
                "INSERT INTO results (image_path, answers) VALUES (?, ?)",
                (result['image_path'], json.dumps(result['answers']))
            )
        conn.commit()
        conn.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = OMRApp(root)
    root.mainloop()
