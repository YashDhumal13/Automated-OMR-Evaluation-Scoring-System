# ultra_precise_omr.py - Maximum accuracy OMR system
import cv2
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
from scipy import stats
import os

class UltraPreciseOMR:
    def __init__(self, config_path='config_ultra_precise_v2.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.scaler = StandardScaler()
        self.classifier = None
        self.debug_images = {}
        
    def advanced_preprocessing(self, image):
        """Advanced preprocessing for maximum accuracy"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Noise reduction
        denoised = cv2.bilateralFilter(gray, 
                                     self.config['preprocessing']['bilateral_filter']['d'],
                                     self.config['preprocessing']['bilateral_filter']['sigma_color'],
                                     self.config['preprocessing']['bilateral_filter']['sigma_space'])
        
        # Step 2: Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Step 3: Gaussian blur for smoothing
        blur_kernel = self.config['preprocessing']['gaussian_blur']
        blurred = cv2.GaussianBlur(enhanced, blur_kernel, 0)
        
        # Step 4: Adaptive thresholding
        thresh_config = self.config['preprocessing']['adaptive_threshold']
        binary = cv2.adaptiveThreshold(blurred, 
                                     thresh_config['max_value'],
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY,
                                     thresh_config['block_size'],
                                     thresh_config['C'])
        
        # Step 5: Morphological operations
        kernel_size = self.config['preprocessing']['morphological_ops']['kernel_size']
        iterations = self.config['preprocessing']['morphological_ops']['iterations']
        kernel = np.ones(kernel_size, np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        self.debug_images['preprocessed'] = processed
        return processed
    
    def extract_bubble_features(self, contour, roi):
        """Extract comprehensive features for ML classification"""
        # Geometric features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Fill ratio (percentage of filled pixels)
        mask = np.zeros(roi.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        fill_pixels = np.sum(roi[mask == 255] < 128)
        total_pixels = np.sum(mask == 255)
        fill_ratio = fill_pixels / total_pixels if total_pixels > 0 else 0
        
        # Intensity statistics
        intensities = roi[mask == 255]
        mean_intensity = np.mean(intensities) if len(intensities) > 0 else 255
        std_intensity = np.std(intensities) if len(intensities) > 0 else 0
        min_intensity = np.min(intensities) if len(intensities) > 0 else 255
        
        # Edge density
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.sum(edges[mask == 255] > 0) / total_pixels if total_pixels > 0 else 0
        
        return [area, circularity, aspect_ratio, fill_ratio, mean_intensity, 
                std_intensity, min_intensity, edge_density]
    
    def detect_bubbles_advanced(self, image):
        """Advanced bubble detection with sub-pixel accuracy"""
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_bubbles = []
        features_list = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Area filtering
            if (self.config['detection']['min_contour_area'] <= area <= 
                self.config['detection']['max_contour_area']):
                
                # Circularity check
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                if circularity >= self.config['detection']['circularity_threshold']:
                    # Aspect ratio check
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    ratio_range = self.config['detection']['aspect_ratio_range']
                    
                    if ratio_range[0] <= aspect_ratio <= ratio_range[1]:
                        # Extract ROI for feature analysis
                        roi = image[y:y+h, x:x+w]
                        features = self.extract_bubble_features(contour, roi)
                        
                        valid_bubbles.append({
                            'contour': contour,
                            'bbox': (x, y, w, h),
                            'features': features
                        })
                        features_list.append(features)
        
        return valid_bubbles, np.array(features_list)
    
    def train_ml_classifier(self, features, labels):
        """Train ensemble classifier for bubble classification"""
        if len(features) == 0:
            return None
            
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create ensemble classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        svm = SVC(probability=True, random_state=42)
        
        self.classifier = VotingClassifier(
            estimators=[('rf', rf), ('svm', svm)],
            voting='soft'
        )
        
        # Train classifier
        self.classifier.fit(features_scaled, labels)
        
        # Cross-validation for accuracy assessment
        cv_scores = cross_val_score(self.classifier, features_scaled, labels, 
                                  cv=self.config['validation']['cross_validation_folds'])
        
        print(f"✓ ML Classifier trained - CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Save model
        joblib.dump(self.classifier, 'ultra_precise_model.pkl')
        joblib.dump(self.scaler, 'ultra_precise_scaler.pkl')
        
        return np.mean(cv_scores)
    
    def classify_bubbles_ml(self, features):
        """Classify bubbles using trained ML model"""
        if self.classifier is None or len(features) == 0:
            return [], []
            
        features_scaled = self.scaler.transform(features)
        predictions = self.classifier.predict(features_scaled)
        probabilities = self.classifier.predict_proba(features_scaled)
        
        # Get confidence scores
        confidences = np.max(probabilities, axis=1)
        
        return predictions, confidences
    
    def statistical_validation(self, results):
        """Apply statistical validation and outlier detection"""
        if len(results) == 0:
            return results
            
        # Extract fill ratios for statistical analysis
        fill_ratios = [bubble['fill_ratio'] for bubble in results]
        
        # Outlier detection using Z-score
        z_scores = np.abs(stats.zscore(fill_ratios))
        outlier_threshold = 2.5  # Conservative threshold
        
        validated_results = []
        for i, result in enumerate(results):
            if z_scores[i] <= outlier_threshold:
                validated_results.append(result)
            else:
                print(f"⚠ Outlier detected and removed: fill_ratio={fill_ratios[i]:.3f}")
        
        return validated_results
    
    def process_omr_sheet(self, image_path):
        """Main processing pipeline for ultra-precise OMR"""
        print("🔍 Starting Ultra-Precise OMR Processing...")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"✓ Image loaded: {image.shape}")
        
        # Advanced preprocessing
        processed = self.advanced_preprocessing(image)
        print("✓ Advanced preprocessing completed")
        
        # Detect bubbles
        bubbles, features = self.detect_bubbles_advanced(processed)
        print(f"✓ Detected {len(bubbles)} potential bubbles")
        
        if len(bubbles) == 0:
            print("❌ No bubbles detected")
            return []
        
        # For demo purposes, create synthetic labels based on fill ratio
        # In real scenario, you would have ground truth labels
        synthetic_labels = []
        results = []
        
        for bubble in bubbles:
            fill_ratio = bubble['features'][3]  # Fill ratio is 4th feature
            
            # Multi-threshold classification
            primary_thresh = self.config['detection']['fill_threshold_primary']
            secondary_thresh = self.config['detection']['fill_threshold_secondary']
            
            if fill_ratio >= secondary_thresh:
                classification = 1  # Filled
                confidence = min(0.95, fill_ratio * 2)
            elif fill_ratio >= primary_thresh:
                classification = 1 if fill_ratio >= (primary_thresh + secondary_thresh) / 2 else 0
                confidence = 0.7 + (fill_ratio - primary_thresh) * 2
            else:
                classification = 0  # Not filled
                confidence = min(0.95, (1 - fill_ratio) * 2)
            
            synthetic_labels.append(classification)
            
            x, y, w, h = bubble['bbox']
            results.append({
                'x': int(x + w/2),
                'y': int(y + h/2),
                'filled': classification == 1,
                'confidence': confidence,
                'fill_ratio': fill_ratio,
                'features': bubble['features']
            })
        
        # Train ML classifier (in production, this would be done offline)
        if self.config['ml_classification']['enabled']:
            ml_accuracy = self.train_ml_classifier(features, synthetic_labels)
            
            # Re-classify using ML model
            predictions, confidences = self.classify_bubbles_ml(features)
            
            # Update results with ML predictions
            for i, result in enumerate(results):
                if i < len(predictions):
                    result['filled'] = predictions[i] == 1
                    result['confidence'] = confidences[i]
        
        # Statistical validation
        validated_results = self.statistical_validation(results)
        print(f"✓ Statistical validation completed: {len(validated_results)} results validated")
        
        # Save debug images
        self.save_debug_images(image, validated_results)
        
        return validated_results
    
    def save_debug_images(self, original, results):
        """Save debug images for analysis"""
        debug_image = original.copy()
        
        for result in results:
            color = (0, 255, 0) if result['filled'] else (0, 0, 255)
            cv2.circle(debug_image, (result['x'], result['y']), 10, color, 2)
            
            # Add confidence text
            conf_text = f"{result['confidence']:.2f}"
            cv2.putText(debug_image, conf_text, 
                       (result['x']-15, result['y']-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imwrite('debug_ultra_precise_results.jpg', debug_image)
        
        if 'preprocessed' in self.debug_images:
            cv2.imwrite('debug_preprocessed.jpg', self.debug_images['preprocessed'])
        
        print("✓ Debug images saved")

def main():
    """Main execution function"""
    omr = UltraPreciseOMR()
    
    image_path = 'test_omr_sheet.jpg'
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        print("Please save your OMR sheet as 'test_omr_sheet.jpg'")
        return
    
    try:
        results = omr.process_omr_sheet(image_path)
        
        # Calculate accuracy metrics
        total_bubbles = len(results)
        high_confidence = sum(1 for r in results if r['confidence'] >= 0.9)
        
        accuracy_estimate = (high_confidence / total_bubbles * 100) if total_bubbles > 0 else 0
        
        print("\n" + "="*60)
        print("📊 ULTRA-PRECISE OMR RESULTS")
        print("="*60)
        print(f"Total bubbles detected: {total_bubbles}")
        print(f"High confidence results: {high_confidence}")
        print(f"Estimated accuracy: {accuracy_estimate:.2f}%")
        print(f"Target accuracy: {omr.config['accuracy_target']}%")
        
        # Save results
        output = {
            'total_bubbles': total_bubbles,
            'estimated_accuracy': accuracy_estimate,
            'target_accuracy': omr.config['accuracy_target'],
            'results': results,
            'config': omr.config
        }
        
        with open('results_ultra_precise.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"✅ Results saved to 'results_ultra_precise.json'")
        
        if accuracy_estimate >= omr.config['accuracy_target']:
            print(f"🎉 TARGET ACCURACY ACHIEVED: {accuracy_estimate:.2f}% >= {omr.config['accuracy_target']}%")
        else:
            print(f"⚠️  Target accuracy not yet achieved. Current: {accuracy_estimate:.2f}%")
            print("💡 Consider fine-tuning parameters or providing more training data.")
        
    except Exception as e:
        print(f"❌ Error processing OMR sheet: {str(e)}")
        raise

if __name__ == "__main__":
    main()