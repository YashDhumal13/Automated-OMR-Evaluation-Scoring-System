import cv2
import numpy as np
import json
from ultra_precise_omr import UltraPreciseOMR
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class AccuracyBenchmark:
    def __init__(self):
        self.omr_processor = UltraPreciseOMR()
        self.results = {}
    
    def create_synthetic_test_cases(self, n_cases=1000):
        """Create synthetic test cases for accuracy validation"""
        test_cases = []
        
        for i in range(n_cases):
            # Generate synthetic bubble with known ground truth
            # Various fill levels, noise conditions, and distortions
            fill_level = np.random.uniform(0, 1)
            noise_level = np.random.uniform(0, 0.1)
            
            # Create synthetic bubble image
            bubble_img = np.ones((50, 50), dtype=np.uint8) * 255
            center = (25, 25)
            radius = 20
            
            # Draw circle
            cv2.circle(bubble_img, center, radius, 0, 2)
            
            # Fill based on fill_level
            if fill_level > 0.15:  # Threshold for "filled"
                fill_radius = int(radius * 0.8 * fill_level)
                cv2.circle(bubble_img, center, fill_radius, 50, -1)
            
            # Add noise
            noise = np.random.randint(0, int(noise_level * 255), bubble_img.shape)
            bubble_img = np.clip(bubble_img.astype(int) + noise, 0, 255).astype(np.uint8)
            
            ground_truth = 1 if fill_level > 0.15 else 0
            
            test_cases.append({
                'image': bubble_img,
                'ground_truth': ground_truth,
                'fill_level': fill_level,
                'noise_level': noise_level
            })
        
        return test_cases
    
    def run_accuracy_test(self, test_cases):
        """Run accuracy test on synthetic test cases"""
        print("🧪 Running Accuracy Benchmark Test...")
        print(f"📊 Test cases: {len(test_cases)}")
        
        predictions = []
        ground_truths = []
        confidences = []
        
        start_time = time.time()
        
        for i, test_case in enumerate(test_cases):
            if i % 100 == 0:
                print(f"Processing: {i}/{len(test_cases)}")
            
            # Process single bubble
            bubbles, features = self.omr_processor.detect_bubbles_advanced(test_case['image'])
            
            if len(bubbles) > 0:
                # Use the first (and likely only) bubble
                bubble = bubbles[0]
                fill_ratio = bubble['features'][3]  # Fill ratio feature
                
                # Apply thresholding
                prediction = 1 if fill_ratio > 0.15 else 0
                confidence = min(0.95, abs(fill_ratio - 0.15) * 5 + 0.5)
            else:
                prediction = 0
                confidence = 0.1
            
            predictions.append(prediction)
            ground_truths.append(test_case['ground_truth'])
            confidences.append(confidence)
        
        processing_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(ground_truths, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(ground_truths, predictions, average='binary')
        
        # Confusion matrix
        cm = confusion_matrix(ground_truths, predictions)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'processing_time': processing_time,
            'test_cases': len(test_cases),
            'average_confidence': np.mean(confidences)
        }
        
        self.results = results
        return results
    
    def generate_report(self):
        """Generate comprehensive accuracy report"""
        if not self.results:
            print("❌ No benchmark results available")
            return
        
        results = self.results
        
        print("\n" + "="*80)
        print("📈 ULTRA-PRECISE OMR ACCURACY BENCHMARK REPORT")
        print("="*80)
        print(f"🎯 Target Accuracy: 99.5%")
        print(f"📊 Achieved Accuracy: {results['accuracy']*100:.3f}%")
        print(f"✨ Precision: {results['precision']*100:.2f}%")
        print(f"🔍 Recall: {results['recall']*100:.2f}%")
        print(f"⚖️ F1-Score: {results['f1_score']*100:.2f}%")
        print(f"⏱️ Processing Time: {results['processing_time']:.2f} seconds")
        print(f"🏃‍♂️ Speed: {results['test_cases']/results['processing_time']:.1f} bubbles/second")
        print(f"🎪 Average Confidence: {results['average_confidence']*100:.1f}%")
        
        print("\n📊 Confusion Matrix:")
        cm = np.array(results['confusion_matrix'])
        print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
        
        # Accuracy achievement check
        if results['accuracy'] >= 0.995:
            print("\n🎉 SUCCESS! Target accuracy of 99.5% ACHIEVED!")
            print(f"✅ Actual accuracy: {results['accuracy']*100:.3f}%")
        else:
            shortfall = 0.995 - results['accuracy']
            print(f"\n⚠️ Target accuracy not achieved. Shortfall: {shortfall*100:.3f}%")
            print("💡 Recommendations:")
            if results['precision'] < 0.995:
                print("   - Reduce false positives by increasing fill threshold")
            if results['recall'] < 0.995:
                print("   - Reduce false negatives by decreasing fill threshold")
            print("   - Consider ensemble methods or additional features")
        
        # Save report
        with open('accuracy_benchmark_report.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Full report saved to 'accuracy_benchmark_report.json'")
    
    def plot_results(self):
        """Create visualization plots for the benchmark results"""
        if not self.results:
            return
            
        # Create confusion matrix heatmap
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        cm = np.array(self.results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.subplot(1, 2, 2)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [self.results['accuracy'], self.results['precision'], 
                 self.results['recall'], self.results['f1_score']]
        
        bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
        plt.axhline(y=0.995, color='red', linestyle='--', label='Target (99.5%)')
        plt.ylim(0.98, 1.0)
        plt.title('Performance Metrics')
        plt.ylabel('Score')
        plt.legend()
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('benchmark_results_plot.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Run the complete accuracy benchmark"""
    benchmark = AccuracyBenchmark()
    
    print("🚀 Starting Ultra-Precise OMR Accuracy Benchmark")
    print("Target: 99.5% accuracy")
    
    # Create test cases
    test_cases = benchmark.create_synthetic_test_cases(1000)
    
    # Run benchmark
    results = benchmark.run_accuracy_test(test_cases)
    
    # Generate report
    benchmark.generate_report()
    
    # Create plots
    benchmark.plot_results()

if __name__ == "__main__":
    main()
