"""
Bubble Classification Module

This module determines whether each detected bubble is marked or unmarked
by analyzing the pixel density within bubble boundaries.

Key Features:
- Pixel density analysis for marked/unmarked classification
- Adaptive thresholding for different image conditions
- Confidence scoring for classification results
- Support for partial markings and erasures
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bubble_detection import Bubble

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BubbleClassification:
    """Represents the classification result for a single bubble."""
    bubble: Bubble
    is_marked: bool
    confidence: float
    dark_pixel_ratio: float
    dark_pixel_count: int
    total_pixels: int


class BubbleClassifier:
    """
    Classifies bubbles as marked or unmarked based on pixel analysis.
    
    This class analyzes the pixel content within each bubble to determine
    if it has been filled in by the student.
    """
    
    def __init__(self, 
                 marked_threshold: float = 0.3,
                 confidence_threshold: float = 0.7,
                 erosion_kernel_size: int = 3):
        """
        Initialize the bubble classifier.
        
        Args:
            marked_threshold: Minimum ratio of dark pixels to consider a bubble marked
            confidence_threshold: Minimum confidence score for reliable classification
            erosion_kernel_size: Size of erosion kernel for noise reduction
        """
        self.marked_threshold = marked_threshold
        self.confidence_threshold = confidence_threshold
        self.erosion_kernel_size = erosion_kernel_size
    
    def preprocess_for_classification(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for bubble classification.
        
        Args:
            image: Input image (can be grayscale or color)
            
        Returns:
            Binary image suitable for pixel analysis
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply binary threshold using Otsu's method
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological operations to reduce noise
        if self.erosion_kernel_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (self.erosion_kernel_size, self.erosion_kernel_size))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def extract_bubble_region(self, binary_image: np.ndarray, bubble: Bubble) -> np.ndarray:
        """
        Extract the region of interest for a specific bubble.
        
        Args:
            binary_image: Preprocessed binary image
            bubble: Bubble object containing contour information
            
        Returns:
            Binary mask of the bubble region
        """
        # Create a mask for the bubble
        mask = np.zeros(binary_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [bubble.contour], 255)
        
        # Extract the bubble region
        bubble_region = cv2.bitwise_and(binary_image, mask)
        
        return bubble_region
    
    def analyze_pixel_density(self, bubble_region: np.ndarray, bubble: Bubble) -> Tuple[int, int, float]:
        """
        Analyze pixel density within the bubble region.
        
        Args:
            bubble_region: Binary image of the bubble region
            bubble: Bubble object
            
        Returns:
            Tuple of (dark_pixel_count, total_pixels, dark_pixel_ratio)
        """
        # Create a mask for the bubble area
        mask = np.zeros(bubble_region.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [bubble.contour], 255)
        
        # Count total pixels in the bubble
        total_pixels = cv2.countNonZero(mask)
        
        # Count dark pixels (marked areas) in the bubble
        dark_pixels = cv2.bitwise_and(bubble_region, mask)
        dark_pixel_count = cv2.countNonZero(dark_pixels)
        
        # Calculate ratio
        dark_pixel_ratio = dark_pixel_count / total_pixels if total_pixels > 0 else 0
        
        return dark_pixel_count, total_pixels, dark_pixel_ratio
    
    def calculate_confidence(self, dark_pixel_ratio: float, bubble_area: float) -> float:
        """
        Calculate confidence score for the classification.
        
        Args:
            dark_pixel_ratio: Ratio of dark pixels in the bubble
            bubble_area: Area of the bubble
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence on how far the ratio is from the threshold
        distance_from_threshold = abs(dark_pixel_ratio - self.marked_threshold)
        
        # Higher distance means higher confidence
        base_confidence = min(distance_from_threshold * 2, 1.0)
        
        # Adjust confidence based on bubble size (larger bubbles are more reliable)
        size_factor = min(bubble_area / 1000.0, 1.0)  # Normalize to 1000 pixels
        
        # Combine factors
        confidence = base_confidence * (0.7 + 0.3 * size_factor)
        
        return min(confidence, 1.0)
    
    def classify_single_bubble(self, image: np.ndarray, bubble: Bubble) -> BubbleClassification:
        """
        Classify a single bubble as marked or unmarked.
        
        Args:
            image: Preprocessed binary image
            bubble: Bubble to classify
            
        Returns:
            BubbleClassification object with results
        """
        # Extract bubble region
        bubble_region = self.extract_bubble_region(image, bubble)
        
        # Analyze pixel density
        dark_pixel_count, total_pixels, dark_pixel_ratio = self.analyze_pixel_density(
            bubble_region, bubble
        )
        
        # Determine if marked
        is_marked = dark_pixel_ratio >= self.marked_threshold
        
        # Calculate confidence
        confidence = self.calculate_confidence(dark_pixel_ratio, bubble.area)
        
        return BubbleClassification(
            bubble=bubble,
            is_marked=is_marked,
            confidence=confidence,
            dark_pixel_ratio=dark_pixel_ratio,
            dark_pixel_count=dark_pixel_count,
            total_pixels=total_pixels
        )
    
    def classify_bubbles(self, image: np.ndarray, 
                        organized_bubbles: List[List[Bubble]],
                        debug: bool = False) -> Tuple[List[List[BubbleClassification]], dict]:
        """
        Classify all bubbles in the organized bubble structure.
        
        Args:
            image: Input image
            organized_bubbles: List of questions, each containing option bubbles
            debug: Whether to return debug information
            
        Returns:
            Tuple of (classified_bubbles, debug_info)
            classified_bubbles: List of questions with classified bubbles
            debug_info: Dictionary containing processing information
        """
        debug_info = {}
        
        try:
            # Preprocess image
            logger.info("Preprocessing image for bubble classification...")
            binary_image = self.preprocess_for_classification(image)
            if debug:
                debug_info['binary_image'] = binary_image.copy()
            
            # Classify all bubbles
            classified_questions = []
            total_bubbles = 0
            marked_bubbles = 0
            low_confidence_count = 0
            
            for question_bubbles in organized_bubbles:
                classified_question = []
                
                for bubble in question_bubbles:
                    classification = self.classify_single_bubble(binary_image, bubble)
                    classified_question.append(classification)
                    
                    total_bubbles += 1
                    if classification.is_marked:
                        marked_bubbles += 1
                    if classification.confidence < self.confidence_threshold:
                        low_confidence_count += 1
                
                classified_questions.append(classified_question)
            
            # Compile debug information
            if debug:
                debug_info.update({
                    'total_bubbles': total_bubbles,
                    'marked_bubbles': marked_bubbles,
                    'unmarked_bubbles': total_bubbles - marked_bubbles,
                    'low_confidence_count': low_confidence_count,
                    'marked_threshold': self.marked_threshold,
                    'confidence_threshold': self.confidence_threshold
                })
            
            logger.info(f"Classification completed: {marked_bubbles}/{total_bubbles} bubbles marked")
            return classified_questions, debug_info
            
        except Exception as e:
            logger.error(f"Error during bubble classification: {str(e)}")
            debug_info['error'] = str(e)
            return [], debug_info
    
    def get_student_answers(self, classified_bubbles: List[List[BubbleClassification]]) -> Dict[int, str]:
        """
        Extract student answers from classified bubbles.
        
        Args:
            classified_bubbles: List of questions with classified bubbles
            
        Returns:
            Dictionary mapping question numbers to selected answers
        """
        student_answers = {}
        
        for question_classifications in classified_bubbles:
            if not question_classifications:
                continue
            
            question_number = question_classifications[0].bubble.question_number
            marked_options = []
            
            for classification in question_classifications:
                if classification.is_marked and classification.confidence >= self.confidence_threshold:
                    marked_options.append(classification.bubble.option_letter)
            
            # Handle different marking scenarios
            if len(marked_options) == 0:
                student_answers[question_number] = None  # No answer
            elif len(marked_options) == 1:
                student_answers[question_number] = marked_options[0]  # Single answer
            else:
                # Multiple answers - this might be an error or intentional
                student_answers[question_number] = ','.join(sorted(marked_options))
        
        return student_answers
    
    def get_classification_summary(self, classified_bubbles: List[List[BubbleClassification]]) -> Dict:
        """
        Get a summary of the classification results.
        
        Args:
            classified_bubbles: List of questions with classified bubbles
            
        Returns:
            Dictionary with summary statistics
        """
        total_questions = len(classified_bubbles)
        total_bubbles = sum(len(q) for q in classified_bubbles)
        marked_bubbles = sum(sum(1 for c in q if c.is_marked) for q in classified_bubbles)
        
        # Calculate confidence statistics
        all_confidences = [c.confidence for q in classified_bubbles for c in q]
        avg_confidence = np.mean(all_confidences) if all_confidences else 0
        min_confidence = np.min(all_confidences) if all_confidences else 0
        
        # Count questions with answers
        answered_questions = 0
        multiple_answers = 0
        
        for question_classifications in classified_bubbles:
            marked_in_question = sum(1 for c in question_classifications if c.is_marked)
            if marked_in_question > 0:
                answered_questions += 1
            if marked_in_question > 1:
                multiple_answers += 1
        
        return {
            'total_questions': total_questions,
            'total_bubbles': total_bubbles,
            'marked_bubbles': marked_bubbles,
            'answered_questions': answered_questions,
            'unanswered_questions': total_questions - answered_questions,
            'multiple_answers': multiple_answers,
            'average_confidence': avg_confidence,
            'minimum_confidence': min_confidence,
            'marking_rate': marked_bubbles / total_bubbles if total_bubbles > 0 else 0
        }
    
    def visualize_classifications(self, image: np.ndarray, 
                                 classified_bubbles: List[List[BubbleClassification]]) -> np.ndarray:
        """
        Visualize bubble classifications on the image.
        
        Args:
            image: Original image
            classified_bubbles: List of questions with classified bubbles
            
        Returns:
            Image with classification results highlighted
        """
        visualization = image.copy()
        
        for question_classifications in classified_bubbles:
            for classification in question_classifications:
                bubble = classification.bubble
                
                # Choose color based on classification
                if classification.is_marked:
                    if classification.confidence >= self.confidence_threshold:
                        color = (0, 255, 0)  # Green for confident marked
                    else:
                        color = (0, 255, 255)  # Yellow for uncertain marked
                else:
                    if classification.confidence >= self.confidence_threshold:
                        color = (255, 0, 0)  # Red for confident unmarked
                    else:
                        color = (255, 165, 0)  # Orange for uncertain unmarked
                
                # Draw contour
                cv2.drawContours(visualization, [bubble.contour], -1, color, 2)
                
                # Add classification info
                if bubble.question_number and bubble.option_letter:
                    label = f"Q{bubble.question_number}{bubble.option_letter}"
                    if classification.is_marked:
                        label += "✓"
                    
                    cv2.putText(visualization, label,
                              (bubble.center[0] - 20, bubble.center[1] - 15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    # Add confidence score
                    conf_text = f"{classification.confidence:.2f}"
                    cv2.putText(visualization, conf_text,
                              (bubble.center[0] - 15, bubble.center[1] + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return visualization