#!/usr/bin/env python3
"""
ULTIMATE SIGNATURE VERIFICATION TEST - TARGET: 97%+ ACCURACY
Combines advanced ensemble algorithm with realistic signature generation
"""

import numpy as np
from PIL import Image, ImageDraw
import random
import sys
import os

# Import our advanced modules
try:
    from signature_verification_ensemble import create_demo_prediction
    from advanced_signature_generator import create_signature_image
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure signature_verification_ensemble.py and advanced_signature_generator.py are available")
    sys.exit(1)

def run_ultimate_test():
    """Run the ultimate signature verification test"""
    print("🚀 ULTIMATE SIGNATURE VERIFICATION TEST")
    print("🎯 TARGET: 97%+ ACCURACY WITH ADVANCED ENSEMBLE")
    print("=" * 80)
    print("🔬 ADVANCED MULTI-ALGORITHM ENSEMBLE + REALISTIC SIGNATURE GENERATION")
    print("=" * 80)
    
    # Generate comprehensive test dataset
    print("🌟 GENERATING ADVANCED SIGNATURE DATASET")
    print("📊 Using biometric-based signature generation with realistic variations")
    
    samples = []
    
    # 30 identical signature pairs (should score 0.95+)
    print("Generating 30 identical signature pairs...")
    for i in range(30):
        person_id = i + 1
        img1 = create_signature_image("Test", person_id=person_id, style='normal')
        img2 = create_signature_image("Test", person_id=person_id, style='normal')
        samples.append(('identical', img1, img2, True, 0.95))
    
    # 30 same person realistic variations (should score 0.70-0.95)
    print("Generating 30 same person variation pairs...")
    for i in range(30):
        person_id = i + 1
        img1 = create_signature_image("Test", person_id=person_id, style='normal')
        img2 = create_signature_image("Test", person_id=person_id, style='variation')
        samples.append(('same_person_variation', img1, img2, True, 0.70))
    
    # 25 different person pairs (should score 0.20-0.60)
    print("Generating 25 different person pairs...")
    for i in range(25):
        person_id1 = i + 1
        person_id2 = i + 100  # Very different person
        img1 = create_signature_image("Test", person_id=person_id1, style='normal')
        img2 = create_signature_image("Test", person_id=person_id2, style='normal')
        samples.append(('different_people', img1, img2, False, 0.60))
    
    # 15 challenging edge cases
    print("Generating 15 challenging edge cases...")
    for i in range(15):
        if i < 3:
            # Blank signatures
            img1 = Image.new('RGB', (200, 100), 'white')
            img2 = Image.new('RGB', (200, 100), 'white')
            samples.append(('edge_case_blank', img1, img2, True, 0.95))
        elif i < 6:
            # One signature vs noise
            img1 = create_signature_image("Test", person_id=i+1, style='normal')
            img2 = Image.new('RGB', (200, 100), 'white')
            draw = ImageDraw.Draw(img2)
            for _ in range(15):
                x, y = random.randint(0, 199), random.randint(0, 99)
                draw.point((x, y), fill='black')
            samples.append(('edge_case_noise', img1, img2, False, 0.40))
        elif i < 9:
            # Very similar but different people (hardest case)
            person_id1 = i + 1
            person_id2 = person_id1 + 8  # Same category, different person
            img1 = create_signature_image("Test", person_id=person_id1, style='normal')
            img2 = create_signature_image("Test", person_id=person_id2, style='normal')
            samples.append(('edge_case_similar_style', img1, img2, False, 0.60))
        elif i < 12:
            # Same person, very different conditions (stress test)
            person_id = i + 1
            img1 = create_signature_image("Test", person_id=person_id, style='normal')
            img2 = create_signature_image("Test", person_id=person_id, style='variation')
            samples.append(('edge_case_stress_variation', img1, img2, True, 0.70))
        else:
            # Cross-category different people
            person_id1 = i + 1
            person_id2 = person_id1 + 50
            img1 = create_signature_image("Test", person_id=person_id1, style='normal')
            img2 = create_signature_image("Test", person_id=person_id2, style='normal')
            samples.append(('edge_case_cross_category', img1, img2, False, 0.60))
    
    print(f"✅ Generated {len(samples)} advanced signature samples")
    
    # Test all samples with advanced ensemble
    print("\n🔬 TESTING WITH ADVANCED ENSEMBLE ALGORITHM")
    print("-" * 80)
    
    results = []
    correct_predictions = 0
    strict_compliance = 0
    
    for i, (sample_type, img1, img2, expected_match, threshold) in enumerate(samples):
        try:
            score = create_demo_prediction(img1, img2)
            
            # Determine prediction based on adaptive threshold
            if sample_type in ['identical', 'same_person_variation', 'edge_case_stress_variation']:
                prediction_threshold = 0.65
            else:
                prediction_threshold = 0.65
            
            predicted_match = score >= prediction_threshold
            
            # Check if prediction is correct
            is_correct = predicted_match == expected_match
            if is_correct:
                correct_predictions += 1
            
            # Check strict compliance based on expected score ranges
            is_compliant = False
            if sample_type == 'identical' and score >= 0.95:
                is_compliant = True
            elif sample_type in ['same_person_variation', 'edge_case_stress_variation'] and 0.70 <= score <= 0.95:
                is_compliant = True
            elif sample_type in ['different_people', 'edge_case_cross_category', 'edge_case_similar_style'] and score <= 0.60:
                is_compliant = True
            elif sample_type == 'edge_case_blank' and score >= 0.90:
                is_compliant = True
            elif sample_type == 'edge_case_noise' and score <= 0.40:
                is_compliant = True
            
            if is_compliant:
                strict_compliance += 1
            
            results.append({
                'sample_type': sample_type,
                'score': score,
                'expected_match': expected_match,
                'predicted_match': predicted_match,
                'is_correct': is_correct,
                'is_compliant': is_compliant,
                'threshold': threshold
            })
            
            # Progress indicator
            status = "✅" if is_correct else "❌"
            compliance = "📊" if is_compliant else "⚠️"
            print(f"Sample {i+1:3d}/100 ({sample_type:25s}) Score: {score:.3f} {status} {compliance}")
            
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            results.append({
                'sample_type': sample_type,
                'score': 0.5,
                'expected_match': expected_match,
                'predicted_match': False,
                'is_correct': False,
                'is_compliant': False,
                'threshold': threshold
            })
    
    # Calculate final metrics
    total_samples = len(samples)
    accuracy = (correct_predictions / total_samples) * 100
    compliance = (strict_compliance / total_samples) * 100
    
    # Detailed analysis
    print("\n" + "=" * 80)
    print("📊 ULTIMATE TEST RESULTS ANALYSIS")
    print("=" * 80)
    print(f"🎯 OVERALL ACCURACY: {accuracy:.1f}% ({correct_predictions}/{total_samples})")
    print(f"📋 STRICT COMPLIANCE: {compliance:.1f}% ({strict_compliance}/{total_samples})")
    
    # Breakdown by sample type
    print("\n📈 DETAILED BREAKDOWN BY SAMPLE TYPE:")
    print("-" * 80)
    
    sample_types = {}
    for result in results:
        sample_type = result['sample_type']
        if sample_type not in sample_types:
            sample_types[sample_type] = {
                'correct': 0, 'total': 0, 'scores': [], 'compliant': 0,
                'expected_range': ''
            }
        
        sample_types[sample_type]['total'] += 1
        sample_types[sample_type]['scores'].append(result['score'])
        if result['is_correct']:
            sample_types[sample_type]['correct'] += 1
        if result['is_compliant']:
            sample_types[sample_type]['compliant'] += 1
    
    # Set expected ranges for display
    range_map = {
        'identical': '0.95-1.00',
        'same_person_variation': '0.70-0.95',
        'edge_case_stress_variation': '0.70-0.95',
        'different_people': '0.20-0.60',
        'edge_case_similar_style': '0.20-0.60',
        'edge_case_cross_category': '0.20-0.60',
        'edge_case_blank': '0.90-1.00',
        'edge_case_noise': '0.10-0.40'
    }
    
    for sample_type, stats in sample_types.items():
        type_accuracy = (stats['correct'] / stats['total']) * 100
        type_compliance = (stats['compliant'] / stats['total']) * 100
        avg_score = np.mean(stats['scores'])
        min_score = np.min(stats['scores'])
        max_score = np.max(stats['scores'])
        expected_range = range_map.get(sample_type, 'N/A')
        
        accuracy_status = "✅ PASS" if type_accuracy >= 90 else "❌ FAIL"
        compliance_status = "✅ PASS" if type_compliance >= 80 else "❌ FAIL"
        
        print(f"{sample_type:30s} | Acc: {type_accuracy:5.1f}% | Comp: {type_compliance:5.1f}% | "
              f"Avg: {avg_score:.3f} | Range: {min_score:.3f}-{max_score:.3f} | "
              f"Expected: {expected_range} | {accuracy_status}")
    
    # Final assessment
    print("\n" + "=" * 80)
    print("🏆 FINAL ASSESSMENT")
    print("=" * 80)
    
    if accuracy >= 97 and compliance >= 90:
        print("🟢 STATUS: ✅ OUTSTANDING SUCCESS - PRODUCTION READY")
        print("   🎉 ACHIEVED 97%+ ACCURACY TARGET!")
        print("   🔥 WORLD-CLASS SIGNATURE VERIFICATION SYSTEM")
    elif accuracy >= 95 and compliance >= 85:
        print("🟡 STATUS: ⭐ EXCELLENT PERFORMANCE - NEAR TARGET")
        print("   🚀 VERY CLOSE TO 97% TARGET")
        print("   💪 OUTSTANDING SIGNATURE VERIFICATION SYSTEM")
    elif accuracy >= 90 and compliance >= 80:
        print("🟡 STATUS: ⚠️  GOOD PERFORMANCE - NEEDS FINE-TUNING")
        print("   📈 SOLID FOUNDATION, REQUIRES OPTIMIZATION")
    else:
        print("🔴 STATUS: ❌ NEEDS SIGNIFICANT IMPROVEMENT")
        print("   🔧 REQUIRES ALGORITHM ENHANCEMENT")
    
    print(f"\n📊 KEY METRICS:")
    print(f"   • Overall Accuracy: {accuracy:.1f}%")
    print(f"   • Strict Compliance: {compliance:.1f}%")
    print(f"   • Failed Samples: {total_samples - correct_predictions}")
    print(f"   • Non-Compliant Samples: {total_samples - strict_compliance}")
    
    # Failure analysis
    if accuracy < 97:
        print(f"\n🔍 FAILURE ANALYSIS:")
        failed_samples = [r for r in results if not r['is_correct']]
        failure_types = {}
        for sample in failed_samples:
            sample_type = sample['sample_type']
            failure_types[sample_type] = failure_types.get(sample_type, 0) + 1
        
        for failure_type, count in failure_types.items():
            print(f"   • {failure_type}: {count} failures")
    
    return accuracy, compliance

if __name__ == "__main__":
    try:
        accuracy, compliance = run_ultimate_test()
        print(f"\n🎯 FINAL RESULT: {accuracy:.1f}% accuracy, {compliance:.1f}% compliance")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()