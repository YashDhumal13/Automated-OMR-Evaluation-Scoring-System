"""
Perspective Correction Module

This module handles the "scanner effect" - transforming skewed, photographed OMR sheets
into perfectly flat, top-down views using edge detection and perspective warping.

Key Features:
- Noise reduction through grayscale conversion and blurring
- Edge detection to find sheet boundaries
- Four-point detection for document corners
- Perspective transformation to create flat scanned appearance
"""

import cv2
import numpy as np
import imutils
from typing import Tuple, Optional, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerspectiveCorrector:
    """
    Handles perspective correction for OMR sheets.
    
    This class takes a photographed OMR sheet and transforms it into a flat,
    top-down view suitable for bubble detection and analysis.
    """
    
    def __init__(self, blur_kernel_size: int = 5, canny_low: int = 75, canny_high: int = 200):
        """
        Initialize the perspective corrector.
        
        Args:
            blur_kernel_size: Size of the Gaussian blur kernel for noise reduction
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
        """
        self.blur_kernel_size = blur_kernel_size
        self.canny_low = canny_low
        self.canny_high = canny_high
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for edge detection.
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed grayscale image ready for edge detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        return blurred
    
    def detect_edges(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """
        Detect edges in the preprocessed image.
        
        Args:
            preprocessed_image: Grayscale, blurred image
            
        Returns:
            Binary edge image
        """
        # Apply Canny edge detection
        edges = cv2.Canny(preprocessed_image, self.canny_low, self.canny_high)
        
        # Apply morphological operations to close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def find_document_contour(self, edges: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the largest four-sided contour in the edge image.
        
        Args:
            edges: Binary edge image
            
        Returns:
            Four-point contour representing the document corners, or None if not found
        """
        # Find all contours
        contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Look for the largest contour with 4 points (quadrilateral)
        for contour in contours[:10]:  # Check top 10 largest contours
            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If we found a 4-point contour, assume it's the document
            if len(approx) == 4:
                return approx
        
        logger.warning("Could not find a 4-point document contour")
        return None
    
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in the format: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            pts: Array of 4 points
            
        Returns:
            Ordered array of points
        """
        # Initialize ordered coordinates
        rect = np.zeros((4, 2), dtype="float32")
        
        # Sum and difference of coordinates
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Top-left point has smallest sum
        rect[0] = pts[np.argmin(s)]
        
        # Bottom-right point has largest sum
        rect[2] = pts[np.argmax(s)]
        
        # Top-right point has smallest difference
        rect[1] = pts[np.argmin(diff)]
        
        # Bottom-left point has largest difference
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def calculate_dimensions(self, ordered_points: np.ndarray) -> Tuple[int, int]:
        """
        Calculate the dimensions of the corrected image.
        
        Args:
            ordered_points: Four ordered corner points
            
        Returns:
            Width and height of the corrected image
        """
        (tl, tr, br, bl) = ordered_points
        
        # Calculate width
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        # Calculate height
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        return max_width, max_height
    
    def apply_perspective_transform(self, image: np.ndarray, ordered_points: np.ndarray) -> np.ndarray:
        """
        Apply perspective transformation to create a flat, top-down view.
        
        Args:
            image: Original input image
            ordered_points: Four ordered corner points of the document
            
        Returns:
            Perspective-corrected image
        """
        # Calculate dimensions
        max_width, max_height = self.calculate_dimensions(ordered_points)
        
        # Define destination points for the perspective transform
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(ordered_points, dst)
        
        # Apply the perspective transform
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        
        return warped
    
    def correct_perspective(self, image: np.ndarray, debug: bool = False) -> Tuple[Optional[np.ndarray], dict]:
        """
        Complete perspective correction pipeline.
        
        Args:
            image: Input BGR image of the OMR sheet
            debug: Whether to return debug information
            
        Returns:
            Tuple of (corrected_image, debug_info)
            corrected_image: Perspective-corrected image or None if failed
            debug_info: Dictionary containing processing steps and metadata
        """
        debug_info = {}
        
        try:
            # Step 1: Preprocess image
            logger.info("Preprocessing image...")
            preprocessed = self.preprocess_image(image)
            if debug:
                debug_info['preprocessed'] = preprocessed.copy()
            
            # Step 2: Detect edges
            logger.info("Detecting edges...")
            edges = self.detect_edges(preprocessed)
            if debug:
                debug_info['edges'] = edges.copy()
            
            # Step 3: Find document contour
            logger.info("Finding document contour...")
            document_contour = self.find_document_contour(edges)
            if document_contour is None:
                logger.error("Failed to find document contour")
                debug_info['error'] = "Document contour not found"
                return None, debug_info
            
            if debug:
                debug_info['document_contour'] = document_contour.copy()
            
            # Step 4: Order points
            ordered_points = self.order_points(document_contour.reshape(4, 2))
            if debug:
                debug_info['ordered_points'] = ordered_points.copy()
            
            # Step 5: Apply perspective transform
            logger.info("Applying perspective transform...")
            corrected_image = self.apply_perspective_transform(image, ordered_points)
            
            if debug:
                debug_info['final_dimensions'] = corrected_image.shape[:2]
            
            logger.info("Perspective correction completed successfully")
            return corrected_image, debug_info
            
        except Exception as e:
            logger.error(f"Error during perspective correction: {str(e)}")
            debug_info['error'] = str(e)
            return None, debug_info
    
    def visualize_detection(self, image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """
        Visualize the detected document contour on the original image.
        
        Args:
            image: Original image
            contour: Detected document contour
            
        Returns:
            Image with contour drawn
        """
        visualization = image.copy()
        cv2.drawContours(visualization, [contour], -1, (0, 255, 0), 3)
        
        # Draw corner points
        for point in contour.reshape(4, 2):
            cv2.circle(visualization, tuple(point.astype(int)), 10, (255, 0, 0), -1)
        
        return visualization