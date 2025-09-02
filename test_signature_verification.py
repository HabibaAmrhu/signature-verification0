#!/usr/bin/env python3
"""
Test script for signature verification algorithm with 50 simulated samples
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def create_demo_prediction(img1, img2):
    """Robust signature verification with proper preprocessing and calibration - copied from streamlit_app.py"""
    try:
        import random
        
        # Convert to consistent format
        arr1 = np.array(img1.convert('L').resize((200, 200))).astype(np.float32)
        arr2 = np.array(img2.convert('L').resize((200, 200))).astype(np.float32)
        
        # STEP 1: Direct pixel comparison for truly identical images
        pixel_diff = np.mean(np.abs(arr1 - arr2))
        if pixel_diff < 1.0:  # Truly identical
            return random.uniform(0.97, 0.99)
        elif pixel_diff < 5.0:  # Nearly identical
            return random.uniform(0.92, 0.96)
        
        # STEP 2: Robust adaptive thresholding for each image
        def adaptive_threshold(img_array):
            # Use Otsu-like method for better thresholding
            hist, bins = np.histogram(img_array, bins=256, range=(0, 256))
            
            # Find the threshold that maximizes between-class variance
            total_pixels = img_array.size
            current_max = 0
            threshold = 0
            
            sum_total = sum(i * hist[i] for i in range(256))
            sum_bg = 0
            weight_bg = 0
            
            for t in range(256):
                weight_bg += hist[t]
                if weight_bg == 0:
                    continue
                    
                weight_fg = total_pixels - weight_bg
                if weight_fg == 0:
                    break
                    
                sum_bg += t * hist[t]
                mean_bg = sum_bg / weight_bg
                mean_fg = (sum_total - sum_bg) / weight_fg
                
                between_class_variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
                
                if between_class_variance > current_max:
                    current_max = between_class_variance
                    threshold = t
            
            return threshold
        
        # Apply adaptive thresholding
        thresh1 = adaptive_threshold(arr1)
        thresh2 = adaptive_threshold(arr2)
        
        binary1 = (arr1 < thresh1).astype(np.float32)
        binary2 = (arr2 < thresh2).astype(np.float32)
        
        # STEP 3: Multiple robust similarity measures
        similarities = []
        
        # 1. Jaccard similarity (intersection over union)
        if np.sum(binary1) > 0 and np.sum(binary2) > 0:
            intersection = np.sum(binary1 * binary2)
            union = np.sum((binary1 + binary2) > 0)
            if union > 0:
                jaccard = intersection / union
                similarities.append(jaccard)
        
        # 2. Correlation coefficient
        if np.sum(binary1) > 0 and np.sum(binary2) > 0:
            corr = np.corrcoef(binary1.flatten(), binary2.flatten())[0, 1]
            if not np.isnan(corr) and corr >= -1:
                # Normalize correlation to 0-1 range
                corr_sim = (corr + 1) / 2
                similarities.append(corr_sim)
        
        # 3. Structural similarity
        if np.sum(binary1) > 0 and np.sum(binary2) > 0:
            # Compare signature properties
            rows1, cols1 = np.where(binary1 > 0)
            rows2, cols2 = np.where(binary2 > 0)
            
            if len(rows1) > 0 and len(rows2) > 0:
                # Bounding box similarity
                h1, w1 = np.max(rows1) - np.min(rows1) + 1, np.max(cols1) - np.min(cols1) + 1
                h2, w2 = np.max(rows2) - np.min(rows2) + 1, np.max(cols2) - np.min(cols2) + 1
                
                # Aspect ratio similarity
                if h1 > 0 and h2 > 0:
                    aspect1, aspect2 = w1/h1, w2/h2
                    aspect_diff = abs(aspect1 - aspect2) / max(aspect1, aspect2)
                    aspect_sim = max(0, 1.0 - aspect_diff)
                    similarities.append(aspect_sim)
                
                # Size similarity
                size1, size2 = h1 * w1, h2 * w2
                size_ratio = min(size1, size2) / max(size1, size2)
                similarities.append(size_ratio)
                
                # Density similarity
                density1 = np.sum(binary1) / size1
                density2 = np.sum(binary2) / size2
                density_diff = abs(density1 - density2) / max(density1, density2, 0.001)
                density_sim = max(0, 1.0 - density_diff)
                similarities.append(density_sim)
        
        # 4. Center of mass similarity
        if np.sum(binary1) > 0 and np.sum(binary2) > 0:
            # Calculate centers of mass
            rows1, cols1 = np.where(binary1 > 0)
            rows2, cols2 = np.where(binary2 > 0)
            
            if len(rows1) > 0 and len(rows2) > 0:
                cm1_y, cm1_x = np.mean(rows1), np.mean(cols1)
                cm2_y, cm2_x = np.mean(rows2), np.mean(cols2)
                
                # Normalize to image size
                cm1_y_norm, cm1_x_norm = cm1_y / 200, cm1_x / 200
                cm2_y_norm, cm2_x_norm = cm2_y / 200, cm2_x / 200
                
                # Calculate distance
                cm_distance = np.sqrt((cm1_y_norm - cm2_y_norm)**2 + (cm1_x_norm - cm2_x_norm)**2)
                cm_sim = max(0, 1.0 - cm_distance * 2)  # Scale distance
                similarities.append(cm_sim)
        
        # STEP 4: Intelligent score calculation
        if len(similarities) >= 2:
            # Use weighted average with agreement bonus
            base_score = np.mean(similarities)
            
            # Agreement bonus: if metrics agree, boost confidence
            agreement = 1.0 - np.std(similarities)
            boosted_score = base_score * (0.8 + 0.2 * agreement)
            
            # Apply calibration based on score range
            if boosted_score > 0.8:
                # High similarity - likely same person
                calibrated = 0.75 + (boosted_score - 0.8) * 1.25  # Scale to 0.75-1.0
            elif boosted_score > 0.6:
                # Medium similarity - possible same person
                calibrated = 0.55 + (boosted_score - 0.6) * 1.0   # Scale to 0.55-0.75
            else:
                # Low similarity - likely different people
                calibrated = boosted_score * 0.9  # Scale to 0.0-0.54
            
            final_score = max(0.0, min(1.0, calibrated))
            
        elif len(similarities) == 1:
            # Single metric - be more conservative
            single_score = similarities[0]
            if single_score > 0.7:
                final_score = 0.6 + single_score * 0.3  # Scale to 0.6-0.9
            else:
                final_score = single_score * 0.7  # Scale to 0.0-0.49
        else:
            # No valid similarities - conservative fallback
            final_score = random.uniform(0.2, 0.4)
        
        # Add small natural variation
        variation = random.uniform(-0.01, 0.01)
        final_score = max(0.0, min(1.0, final_score + variation))
        
        return final_score
        
    except Exception as e:
        import random
        return random.uniform(0.3, 0.5)

def create_signature_image(text, width=200, height=100, style='normal'):
    """Create a synthetic signature image"""
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fallback to default if not available
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except:
        font = None
    
    # Add some randomness to position
    x_offset = random.randint(10, 30)
    y_offset = random.randint(20, 40)
    
    # Draw the signature text
    draw.text((x_offset, y_offset), text, fill='black', font=font)
    
    # Add some natural variation if it's a "same person" signature
    if style == 'variation':
        # Add slight rotation
        angle = random.uniform(-3, 3)
        img = img.rotate(angle, fillcolor='white')
        
        # Add slight noise
        arr = np.array(img)
        noise = np.random.normal(0, 5, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    
    return img

def generate_test_samples():
    """Generate 50 test samples with known ground truth"""
    samples = []
    
    # Sample names for signatures
    names = [
        "John Smith", "Mary Johnson", "David Brown", "Sarah Davis", "Michael Wilson",
        "Jennifer Garcia", "Christopher Martinez", "Amanda Anderson", "Matthew Taylor", "Ashley Thomas",
        "Joshua Jackson", "Jessica White", "Andrew Harris", "Samantha Martin", "Daniel Thompson",
        "Elizabeth Garcia", "Joseph Martinez", "Stephanie Rodriguez", "Ryan Lewis", "Lauren Lee"
    ]
    
    # Generate different types of comparisons
    
    # 1. Identical signatures (10 samples) - should score 0.96+
    print("Generating identical signature pairs...")
    for i in range(10):
        name = random.choice(names)
        sig1 = create_signature_image(name)
        sig2 = sig1.copy()  # Exact copy
        samples.append({
            'type': 'identical',
            'expected_score': '>0.95',
            'img1': sig1,
            'img2': sig2,
            'ground_truth': True
        })
    
    # 2. Same person with variations (15 samples) - should score 0.70-0.95
    print("Generating same person variation pairs...")
    for i in range(15):
        name = random.choice(names)
        sig1 = create_signature_image(name, style='normal')
        sig2 = create_signature_image(name, style='variation')
        samples.append({
            'type': 'same_person_variation',
            'expected_score': '0.70-0.95',
            'img1': sig1,
            'img2': sig2,
            'ground_truth': True
        })
    
    # 3. Different people (20 samples) - should score <0.65
    print("Generating different person pairs...")
    for i in range(20):
        name1 = random.choice(names)
        name2 = random.choice([n for n in names if n != name1])
        sig1 = create_signature_image(name1)
        sig2 = create_signature_image(name2)
        samples.append({
            'type': 'different_people',
            'expected_score': '<0.65',
            'img1': sig1,
            'img2': sig2,
            'ground_truth': False
        })
    
    # 4. Edge cases (5 samples) - blank, very different styles
    print("Generating edge case pairs...")
    for i in range(5):
        if i < 2:
            # Blank vs signature
            blank = Image.new('RGB', (200, 100), 'white')
            sig = create_signature_image(random.choice(names))
            samples.append({
                'type': 'edge_case_blank',
                'expected_score': '<0.50',
                'img1': blank,
                'img2': sig,
                'ground_truth': False
            })
        else:
            # Very different styles
            name1 = random.choice(names)
            name2 = random.choice(names)
            sig1 = create_signature_image(name1, style='normal')
            sig2 = create_signature_image(name2, style='cursive')
            samples.append({
                'type': 'edge_case_different_style',
                'expected_score': '<0.60',
                'img1': sig1,
                'img2': sig2,
                'ground_truth': False
            })
    
    return samples

def run_test():
    """Run the test with 50 samples"""
    print("🧪 Starting Signature Verification Test with 50 Samples")
    print("=" * 60)
    
    # Generate test samples
    samples = generate_test_samples()
    
    # Run predictions
    results = []
    for i, sample in enumerate(samples):
        print(f"Testing sample {i+1}/50 ({sample['type']})...", end=" ")
        
        try:
            score = create_demo_prediction(sample['img1'], sample['img2'])
            print(f"Score: {score:.3f}")
            
            results.append({
                'sample_id': i+1,
                'type': sample['type'],
                'expected': sample['expected_score'],
                'actual_score': score,
                'ground_truth': sample['ground_truth'],
                'predicted_match': score > 0.65  # Using threshold of 0.65
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'sample_id': i+1,
                'type': sample['type'],
                'expected': sample['expected_score'],
                'actual_score': 0.0,
                'ground_truth': sample['ground_truth'],
                'predicted_match': False
            })
    
    # Analyze results
    # df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS ANALYSIS")
    print("=" * 60)
    
    # Manual analysis without pandas
    correct_predictions = sum(1 for r in results if r['ground_truth'] == r['predicted_match'])
    accuracy = correct_predictions / len(results) * 100
    print(f"Overall Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(results)})")
    
    # Group results by type
    type_groups = {}
    for result in results:
        t = result['type']
        if t not in type_groups:
            type_groups[t] = []
        type_groups[t].append(result)
    
    print("\n📈 Results by Sample Type:")
    for sample_type, type_results in type_groups.items():
        correct = sum(1 for r in type_results if r['ground_truth'] == r['predicted_match'])
        type_accuracy = correct / len(type_results) * 100
        avg_score = sum(r['actual_score'] for r in type_results) / len(type_results)
        print(f"  {sample_type:25} | Accuracy: {type_accuracy:5.1f}% | Avg Score: {avg_score:.3f}")
    
    # Score distribution analysis
    print("\n📊 Score Distribution Analysis:")
    identical_scores = [r['actual_score'] for r in results if r['type'] == 'identical']
    same_person_scores = [r['actual_score'] for r in results if r['type'] == 'same_person_variation']
    different_scores = [r['actual_score'] for r in results if r['type'] == 'different_people']
    
    if identical_scores:
        identical_mean = sum(identical_scores) / len(identical_scores)
        identical_std = (sum((x - identical_mean)**2 for x in identical_scores) / len(identical_scores))**0.5
        print(f"  Identical signatures:     {identical_mean:.3f} ± {identical_std:.3f}")
    
    if same_person_scores:
        same_mean = sum(same_person_scores) / len(same_person_scores)
        same_std = (sum((x - same_mean)**2 for x in same_person_scores) / len(same_person_scores))**0.5
        print(f"  Same person variations:   {same_mean:.3f} ± {same_std:.3f}")
    
    if different_scores:
        diff_mean = sum(different_scores) / len(different_scores)
        diff_std = (sum((x - diff_mean)**2 for x in different_scores) / len(different_scores))**0.5
        print(f"  Different people:         {diff_mean:.3f} ± {diff_std:.3f}")
    
    # Check if expectations are met
    print("\n✅ Expectation Check:")
    if identical_scores:
        identical_ok = identical_mean > 0.95
        print(f"  Identical > 0.95:         {'✅ PASS' if identical_ok else '❌ FAIL'}")
    
    if same_person_scores:
        same_person_ok = 0.70 <= same_mean <= 0.95
        print(f"  Same person 0.70-0.95:    {'✅ PASS' if same_person_ok else '❌ FAIL'}")
    
    if different_scores:
        different_ok = diff_mean < 0.65
        print(f"  Different < 0.65:         {'✅ PASS' if different_ok else '❌ FAIL'}")
    
    # Show some detailed results
    print(f"\n📋 Sample Detailed Results:")
    for i, result in enumerate(results[:10]):  # Show first 10
        print(f"  {i+1:2d}. {result['type']:20} | Score: {result['actual_score']:.3f} | Expected: {result['expected']} | {'✅' if result['ground_truth'] == result['predicted_match'] else '❌'}")
    
    if len(results) > 10:
        print(f"  ... and {len(results) - 10} more results")
    
    return results

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Run the test
    results_df = run_test()