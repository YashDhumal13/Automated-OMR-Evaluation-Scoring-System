"""
OMR Processing Pipeline

This module provides the main pipeline that combines all image processing components
into a single, clean interface for processing OMR sheets from raw photos to final scores.

Key Features:
- Complete end-to-end processing pipeline
- Structured JSON output format
- Comprehensive error handling and logging
- Debug mode for troubleshooting
- Batch processing capabilities
"""

import cv2
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
from datetime import datetime

from modules.perspective_correction import PerspectiveCorrector
from modules.bubble_detection import BubbleDetector
from modules.bubble_classification import BubbleClassifier
from modules.scoring import ScoreCalculator, ScoringConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OMRProcessor:
    """
    Main OMR processing pipeline that orchestrates all components.
    
    This class provides a clean interface for processing OMR sheets from
    raw photographed images to final scored results in JSON format.
    """
    
    def __init__(self, 
                 scoring_config: Optional[ScoringConfig] = None,
                 perspective_params: Optional[Dict] = None,
                 bubble_detection_params: Optional[Dict] = None,
                 classification_params: Optional[Dict] = None):
        """
        Initialize the OMR processor with all components.
        
        Args:
            scoring_config: Configuration for scoring system
            perspective_params: Parameters for perspective correction
            bubble_detection_params: Parameters for bubble detection
            classification_params: Parameters for bubble classification
        """
        # Initialize components with custom parameters
        self.perspective_corrector = PerspectiveCorrector(
            **(perspective_params or {})
        )
        
        self.bubble_detector = BubbleDetector(
            **(bubble_detection_params or {})
        )
        
        self.bubble_classifier = BubbleClassifier(
            **(classification_params or {})
        )
        
        self.score_calculator = ScoreCalculator(
            scoring_config or ScoringConfig()
        )
        
        logger.info("OMR Processor initialized successfully")
    
    def load_answer_key(self, answer_key_path: str, sheet_name: Optional[str] = None) -> bool:
        """
        Load the answer key from an Excel file.
        
        Args:
            answer_key_path: Path to the Excel file containing the answer key
            sheet_name: Name of the sheet to read (optional)
            
        Returns:
            True if successful, False otherwise
        """
        return self.score_calculator.load_answer_key_from_excel(answer_key_path, sheet_name)
    
    def set_answer_key(self, answer_key: Dict[int, str], 
                      subject_mapping: Optional[Dict[int, str]] = None):
        """
        Set the answer key manually.
        
        Args:
            answer_key: Dictionary mapping question numbers to correct answers
            subject_mapping: Optional dictionary mapping question numbers to subjects
        """
        self.score_calculator.set_answer_key_manually(answer_key, subject_mapping)
    
    def process_single_image(self, 
                           image_path: str,
                           student_id: Optional[str] = None,
                           questions_per_row: int = 5,
                           options_per_question: int = 4,
                           debug: bool = False) -> Dict[str, Any]:
        """
        Process a single OMR sheet image through the complete pipeline.
        
        Args:
            image_path: Path to the image file
            student_id: Optional student identifier
            questions_per_row: Number of questions per row in the OMR sheet
            options_per_question: Number of options per question (A, B, C, D)
            debug: Whether to include debug information in the output
            
        Returns:
            Dictionary containing the complete processing results in JSON format
        """
        result = {
            "processing_info": {
                "image_path": image_path,
                "student_id": student_id,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error_message": None
            },
            "image_analysis": {},
            "student_answers": {},
            "scoring_results": {},
            "debug_info": {} if debug else None
        }
        
        try:
            # Step 1: Load and validate image
            logger.info(f"Processing image: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            result["image_analysis"]["original_dimensions"] = image.shape[:2]
            
            # Step 2: Perspective correction
            logger.info("Step 1/4: Applying perspective correction...")
            corrected_image, perspective_debug = self.perspective_corrector.correct_perspective(
                image, debug=debug
            )
            
            if corrected_image is None:
                raise RuntimeError("Perspective correction failed")
            
            result["image_analysis"]["corrected_dimensions"] = corrected_image.shape[:2]
            if debug:
                result["debug_info"]["perspective_correction"] = perspective_debug
            
            # Step 3: Bubble detection
            logger.info("Step 2/4: Detecting bubbles...")
            organized_bubbles, detection_debug = self.bubble_detector.detect_bubbles(
                corrected_image, questions_per_row, options_per_question, debug=debug
            )
            
            if not organized_bubbles:
                raise RuntimeError("No bubbles detected")
            
            bubble_info = self.bubble_detector.get_bubble_grid_info(organized_bubbles)
            result["image_analysis"]["bubble_detection"] = bubble_info
            if debug:
                result["debug_info"]["bubble_detection"] = detection_debug
            
            # Step 4: Bubble classification
            logger.info("Step 3/4: Classifying bubble states...")
            classified_bubbles, classification_debug = self.bubble_classifier.classify_bubbles(
                corrected_image, organized_bubbles, debug=debug
            )
            
            # Extract student answers
            student_answers = self.bubble_classifier.get_student_answers(classified_bubbles)
            classification_summary = self.bubble_classifier.get_classification_summary(classified_bubbles)
            
            result["student_answers"] = student_answers
            result["image_analysis"]["classification_summary"] = classification_summary
            if debug:
                result["debug_info"]["bubble_classification"] = classification_debug
            
            # Step 5: Scoring
            logger.info("Step 4/4: Calculating scores...")
            if self.score_calculator.answer_key:
                score_result = self.score_calculator.calculate_score(student_answers, student_id)
                detailed_report = self.score_calculator.generate_detailed_report(score_result)
                result["scoring_results"] = detailed_report
            else:
                logger.warning("No answer key loaded - skipping scoring")
                result["scoring_results"] = {
                    "error": "No answer key loaded",
                    "student_answers_only": True
                }
            
            # Mark as successful
            result["processing_info"]["success"] = True
            logger.info(f"Successfully processed {image_path}")
            
        except Exception as e:
            error_msg = f"Error processing {image_path}: {str(e)}"
            logger.error(error_msg)
            result["processing_info"]["error_message"] = error_msg
            
            if debug:
                import traceback
                result["debug_info"]["exception_traceback"] = traceback.format_exc()
        
        return result
    
    def process_batch_images(self, 
                           image_paths: List[str],
                           student_ids: Optional[List[str]] = None,
                           questions_per_row: int = 5,
                           options_per_question: int = 4,
                           debug: bool = False) -> Dict[str, Any]:
        """
        Process multiple OMR sheet images in batch.
        
        Args:
            image_paths: List of paths to image files
            student_ids: Optional list of student identifiers (must match image_paths length)
            questions_per_row: Number of questions per row in the OMR sheets
            options_per_question: Number of options per question
            debug: Whether to include debug information
            
        Returns:
            Dictionary containing batch processing results
        """
        if student_ids and len(student_ids) != len(image_paths):
            raise ValueError("student_ids length must match image_paths length")
        
        batch_result = {
            "batch_info": {
                "total_images": len(image_paths),
                "successful_processing": 0,
                "failed_processing": 0,
                "timestamp": datetime.now().isoformat()
            },
            "individual_results": {},
            "batch_summary": {}
        }
        
        logger.info(f"Starting batch processing of {len(image_paths)} images")
        
        # Process each image
        for i, image_path in enumerate(image_paths):
            student_id = student_ids[i] if student_ids else f"student_{i+1}"
            
            try:
                result = self.process_single_image(
                    image_path, student_id, questions_per_row, options_per_question, debug
                )
                
                batch_result["individual_results"][student_id] = result
                
                if result["processing_info"]["success"]:
                    batch_result["batch_info"]["successful_processing"] += 1
                else:
                    batch_result["batch_info"]["failed_processing"] += 1
                    
            except Exception as e:
                logger.error(f"Critical error processing {image_path}: {str(e)}")
                batch_result["individual_results"][student_id] = {
                    "processing_info": {
                        "success": False,
                        "error_message": str(e)
                    }
                }
                batch_result["batch_info"]["failed_processing"] += 1
        
        # Generate batch summary
        batch_result["batch_summary"] = self._generate_batch_summary(batch_result)
        
        logger.info(f"Batch processing completed: {batch_result['batch_info']['successful_processing']} successful, "
                   f"{batch_result['batch_info']['failed_processing']} failed")
        
        return batch_result
    
    def _generate_batch_summary(self, batch_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary statistics for batch processing.
        
        Args:
            batch_result: Batch processing results
            
        Returns:
            Dictionary with summary statistics
        """
        successful_results = []
        
        for student_id, result in batch_result["individual_results"].items():
            if result.get("processing_info", {}).get("success", False):
                successful_results.append(result)
        
        if not successful_results:
            return {"message": "No successful processing results to summarize"}
        
        # Calculate statistics
        scores = []
        total_questions_list = []
        
        for result in successful_results:
            scoring = result.get("scoring_results", {})
            if "summary" in scoring:
                scores.append(scoring["summary"]["percentage"])
                total_questions_list.append(
                    scoring.get("statistics", {}).get("total_questions", 0)
                )
        
        summary = {
            "class_statistics": {},
            "common_patterns": {},
            "quality_metrics": {}
        }
        
        if scores:
            summary["class_statistics"] = {
                "average_score": np.mean(scores),
                "highest_score": np.max(scores),
                "lowest_score": np.min(scores),
                "median_score": np.median(scores),
                "standard_deviation": np.std(scores),
                "students_scored": len(scores)
            }
        
        if total_questions_list:
            summary["quality_metrics"] = {
                "average_questions_detected": np.mean(total_questions_list),
                "consistent_detection": np.std(total_questions_list) < 1.0  # Low variance indicates consistency
            }
        
        return summary
    
    def save_results_to_file(self, results: Dict[str, Any], file_path: str) -> bool:
        """
        Save processing results to a JSON file.
        
        Args:
            results: Results dictionary to save
            file_path: Path where to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results to {file_path}: {str(e)}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the OMR processing system.
        
        Returns:
            Dictionary with system information
        """
        return {
            "version": "1.0.0",
            "components": {
                "perspective_correction": {
                    "blur_kernel_size": self.perspective_corrector.blur_kernel_size,
                    "canny_low": self.perspective_corrector.canny_low,
                    "canny_high": self.perspective_corrector.canny_high
                },
                "bubble_detection": {
                    "min_bubble_area": self.bubble_detector.min_bubble_area,
                    "max_bubble_area": self.bubble_detector.max_bubble_area,
                    "min_aspect_ratio": self.bubble_detector.min_aspect_ratio,
                    "max_aspect_ratio": self.bubble_detector.max_aspect_ratio
                },
                "bubble_classification": {
                    "marked_threshold": self.bubble_classifier.marked_threshold,
                    "confidence_threshold": self.bubble_classifier.confidence_threshold
                },
                "scoring": self.score_calculator.get_answer_key_info()
            },
            "supported_formats": {
                "input_images": ["JPEG", "PNG", "BMP", "TIFF"],
                "answer_key": ["Excel (.xlsx, .xls)"],
                "output": ["JSON"]
            }
        }


# Convenience function for simple usage
def process_omr_image(image_path: str, 
                     answer_key_path: Optional[str] = None,
                     student_id: Optional[str] = None,
                     output_path: Optional[str] = None,
                     **kwargs) -> Dict[str, Any]:
    """
    Convenience function to process a single OMR image with minimal setup.
    
    Args:
        image_path: Path to the OMR image
        answer_key_path: Optional path to Excel answer key
        student_id: Optional student identifier
        output_path: Optional path to save results JSON
        **kwargs: Additional parameters for processing
        
    Returns:
        Processing results dictionary
    """
    processor = OMRProcessor()
    
    if answer_key_path:
        processor.load_answer_key(answer_key_path)
    
    results = processor.process_single_image(image_path, student_id, **kwargs)
    
    if output_path:
        processor.save_results_to_file(results, output_path)
    
    return results