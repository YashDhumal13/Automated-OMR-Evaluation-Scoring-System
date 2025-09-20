"""
Bubble Detection Module

This module handles the identification and extraction of answer bubbles from 
perspective-corrected OMR sheets. It finds individual bubbles and organizes
them by question order.

Key Features:
- Binary thresholding for clear bubble boundaries
- Contour detection with size and aspect ratio filtering
- Intelligent bubble sorting by position
- Grid-based organization for structured OMR layouts
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Bubble:
    """Represents a single answer bubble with its properties."""
    contour: np.ndarray
    center: Tuple[int, int]
    bounding_rect: Tuple[int, int, int, int]  # x, y, width, height
    area: float
    question_number: Optional[int] = None
    option_letter: Optional[str] = None
    row: Optional[int] = None
    column: Optional[int] = None


class BubbleDetector:
    """
    Detects and organizes answer bubbles in OMR sheets.
    
    This class takes a perspective-corrected OMR sheet and identifies all
    answer bubbles, organizing them by question and option.
    """
    
    def __init__(self, 
                 min_bubble_area: int = 200,
                 max_bubble_area: int = 2000,
                 min_aspect_ratio: float = 0.5,
                 max_aspect_ratio: float = 2.0,
                 threshold_value: int = 127):
        """
        Initialize the bubble detector.
        
        Args:
            min_bubble_area: Minimum area for a valid bubble
            max_bubble_area: Maximum area for a valid bubble
            min_aspect_ratio: Minimum aspect ratio for a valid bubble
            max_aspect_ratio: Maximum aspect ratio for a valid bubble
            threshold_value: Binary threshold value
        """
        self.min_bubble_area = min_bubble_area
        self.max_bubble_area = max_bubble_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.threshold_value = threshold_value
    
    def preprocess_for_bubbles(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for bubble detection.
        
        Args:
            image: Perspective-corrected OMR sheet
            
        Returns:
            Binary thresholded image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply binary threshold using Otsu's method for automatic threshold selection
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return thresh
    
    def find_bubble_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        Find all potential bubble contours in the binary image.
        
        Args:
            binary_image: Binary thresholded image
            
        Returns:
            List of contours that could be bubbles
        """
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area and aspect ratio
        bubble_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip contours that are too small or too large
            if area < self.min_bubble_area or area > self.max_bubble_area:
                continue
            
            # Calculate bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Check aspect ratio (bubbles should be roughly circular)
            if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                bubble_contours.append(contour)
        
        logger.info(f"Found {len(bubble_contours)} potential bubbles")
        return bubble_contours
    
    def create_bubble_objects(self, contours: List[np.ndarray]) -> List[Bubble]:
        """
        Create Bubble objects from contours.
        
        Args:
            contours: List of bubble contours
            
        Returns:
            List of Bubble objects
        """
        bubbles = []
        
        for contour in contours:
            # Calculate center point
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
            else:
                center_x, center_y = 0, 0
            
            # Get bounding rectangle
            bounding_rect = cv2.boundingRect(contour)
            
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Create bubble object
            bubble = Bubble(
                contour=contour,
                center=(center_x, center_y),
                bounding_rect=bounding_rect,
                area=area
            )
            
            bubbles.append(bubble)
        
        return bubbles
    
    def sort_bubbles_by_position(self, bubbles: List[Bubble], 
                                questions_per_row: int = 5,
                                options_per_question: int = 4) -> List[List[Bubble]]:
        """
        Sort bubbles by their position to organize them by question and option.
        
        Args:
            bubbles: List of detected bubbles
            questions_per_row: Number of questions per row
            options_per_question: Number of options per question (A, B, C, D)
            
        Returns:
            List of lists, where each inner list contains bubbles for one question
        """
        if not bubbles:
            return []
        
        # Sort bubbles by y-coordinate (top to bottom) then x-coordinate (left to right)
        sorted_bubbles = sorted(bubbles, key=lambda b: (b.center[1], b.center[0]))
        
        # Group bubbles into rows based on y-coordinate
        rows = self._group_bubbles_into_rows(sorted_bubbles)
        
        # Process each row to identify questions and options
        organized_bubbles = []
        question_number = 1
        
        for row_bubbles in rows:
            # Sort bubbles in this row by x-coordinate
            row_bubbles.sort(key=lambda b: b.center[0])
            
            # Group bubbles in this row by questions
            questions_in_row = self._group_row_into_questions(row_bubbles, options_per_question)
            
            for question_bubbles in questions_in_row:
                # Assign question number and option letters
                for i, bubble in enumerate(question_bubbles):
                    bubble.question_number = question_number
                    bubble.option_letter = chr(ord('A') + i)  # A, B, C, D, ...
                
                organized_bubbles.append(question_bubbles)
                question_number += 1
        
        logger.info(f"Organized {len(organized_bubbles)} questions")
        return organized_bubbles
    
    def _group_bubbles_into_rows(self, sorted_bubbles: List[Bubble], 
                                row_tolerance: int = 30) -> List[List[Bubble]]:
        """
        Group bubbles into rows based on their y-coordinates.
        
        Args:
            sorted_bubbles: Bubbles sorted by position
            row_tolerance: Tolerance for considering bubbles in the same row
            
        Returns:
            List of bubble rows
        """
        if not sorted_bubbles:
            return []
        
        rows = []
        current_row = [sorted_bubbles[0]]
        current_y = sorted_bubbles[0].center[1]
        
        for bubble in sorted_bubbles[1:]:
            if abs(bubble.center[1] - current_y) <= row_tolerance:
                # Same row
                current_row.append(bubble)
            else:
                # New row
                rows.append(current_row)
                current_row = [bubble]
                current_y = bubble.center[1]
        
        # Add the last row
        if current_row:
            rows.append(current_row)
        
        return rows
    
    def _group_row_into_questions(self, row_bubbles: List[Bubble], 
                                 options_per_question: int) -> List[List[Bubble]]:
        """
        Group bubbles in a row into individual questions.
        
        Args:
            row_bubbles: Bubbles in a single row, sorted by x-coordinate
            options_per_question: Number of options per question
            
        Returns:
            List of questions, each containing a list of option bubbles
        """
        questions = []
        
        # Simple grouping: assume bubbles are evenly spaced and group by options_per_question
        for i in range(0, len(row_bubbles), options_per_question):
            question_bubbles = row_bubbles[i:i + options_per_question]
            if len(question_bubbles) == options_per_question:  # Only include complete questions
                questions.append(question_bubbles)
        
        return questions
    
    def detect_bubbles(self, image: np.ndarray, 
                      questions_per_row: int = 5,
                      options_per_question: int = 4,
                      debug: bool = False) -> Tuple[List[List[Bubble]], dict]:
        """
        Complete bubble detection pipeline.
        
        Args:
            image: Perspective-corrected OMR sheet
            questions_per_row: Number of questions per row
            options_per_question: Number of options per question
            debug: Whether to return debug information
            
        Returns:
            Tuple of (organized_bubbles, debug_info)
            organized_bubbles: List of questions, each containing option bubbles
            debug_info: Dictionary containing processing steps and metadata
        """
        debug_info = {}
        
        try:
            # Step 1: Preprocess image
            logger.info("Preprocessing image for bubble detection...")
            binary_image = self.preprocess_for_bubbles(image)
            if debug:
                debug_info['binary_image'] = binary_image.copy()
            
            # Step 2: Find bubble contours
            logger.info("Finding bubble contours...")
            contours = self.find_bubble_contours(binary_image)
            if debug:
                debug_info['raw_contours_count'] = len(contours)
            
            # Step 3: Create bubble objects
            logger.info("Creating bubble objects...")
            bubbles = self.create_bubble_objects(contours)
            if debug:
                debug_info['bubbles_count'] = len(bubbles)
            
            # Step 4: Sort and organize bubbles
            logger.info("Organizing bubbles by position...")
            organized_bubbles = self.sort_bubbles_by_position(
                bubbles, questions_per_row, options_per_question
            )
            
            if debug:
                debug_info['questions_detected'] = len(organized_bubbles)
                debug_info['bubbles_per_question'] = [len(q) for q in organized_bubbles]
            
            logger.info(f"Bubble detection completed: {len(organized_bubbles)} questions detected")
            return organized_bubbles, debug_info
            
        except Exception as e:
            logger.error(f"Error during bubble detection: {str(e)}")
            debug_info['error'] = str(e)
            return [], debug_info
    
    def visualize_bubbles(self, image: np.ndarray, 
                         organized_bubbles: List[List[Bubble]]) -> np.ndarray:
        """
        Visualize detected bubbles on the original image.
        
        Args:
            image: Original image
            organized_bubbles: Detected and organized bubbles
            
        Returns:
            Image with bubbles highlighted
        """
        visualization = image.copy()
        
        # Color map for different questions
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        for i, question_bubbles in enumerate(organized_bubbles):
            color = colors[i % len(colors)]
            
            for bubble in question_bubbles:
                # Draw contour
                cv2.drawContours(visualization, [bubble.contour], -1, color, 2)
                
                # Draw center point
                cv2.circle(visualization, bubble.center, 3, color, -1)
                
                # Add question and option labels
                if bubble.question_number and bubble.option_letter:
                    label = f"Q{bubble.question_number}{bubble.option_letter}"
                    cv2.putText(visualization, label, 
                              (bubble.center[0] - 15, bubble.center[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return visualization
    
    def get_bubble_grid_info(self, organized_bubbles: List[List[Bubble]]) -> Dict:
        """
        Get information about the detected bubble grid.
        
        Args:
            organized_bubbles: Detected and organized bubbles
            
        Returns:
            Dictionary with grid information
        """
        if not organized_bubbles:
            return {"total_questions": 0, "total_bubbles": 0}
        
        total_questions = len(organized_bubbles)
        total_bubbles = sum(len(question) for question in organized_bubbles)
        options_per_question = len(organized_bubbles[0]) if organized_bubbles else 0
        
        # Calculate grid dimensions
        question_positions = []
        for question_bubbles in organized_bubbles:
            if question_bubbles:
                avg_y = sum(b.center[1] for b in question_bubbles) / len(question_bubbles)
                avg_x = sum(b.center[0] for b in question_bubbles) / len(question_bubbles)
                question_positions.append((avg_x, avg_y))
        
        return {
            "total_questions": total_questions,
            "total_bubbles": total_bubbles,
            "options_per_question": options_per_question,
            "question_positions": question_positions,
            "grid_dimensions": {
                "width": max(pos[0] for pos in question_positions) - min(pos[0] for pos in question_positions) if question_positions else 0,
                "height": max(pos[1] for pos in question_positions) - min(pos[1] for pos in question_positions) if question_positions else 0
            }
        }