#!/usr/bin/env python3
"""
測試 SEM Defect Detector v1 對於具有挑戰性缺陷圖片的檢測能力
"""

import sys
sys.path.append('.')
from sem_defect_detector_v1 import SEMDefectDetector

# 測試缺陷 D6 (在圖片中央，應該比較容易檢測)
# Bbox: (501, 379, 11, 25)
defect_tests = [
    {
        'name': 'D6 - Center defect',
        'bbox': (501, 379, 11, 25),
        'expected': 'BRIGHT'
    },
    {
        'name': 'D1 - Edge defect',
        'bbox': (7, 464, 9, 21),
        'expected': 'BRIGHT'
    },
    {
        'name': 'D8 - Medium brightness',
        'bbox': (474, 477, 9, 22),
        'expected': 'BRIGHT'
    }
]

def test_defect(detector, test_case):
    """測試單個缺陷"""
    print(f"\n{'='*60}")
    print(f"Testing: {test_case['name']}")
    print(f"Target bbox: {test_case['bbox']}")
    print('='*60)
    
    try:
        results = detector.analyze_defect(
            test_case['bbox'],
            edge_margin=50,  # 降低邊界要求，因為圖片較小
            brightness_tolerance=15,  # 提高容差，因為有模糊
            size_tolerance=30  # 提高尺寸容差
        )
        
        print(f"\nDetection Result: {results['defect_type']}")
        print(f"Expected: {test_case['expected']} DEFECT")
        
        if test_case['expected'] in results['defect_type']:
            print("✓ PASS")
        else:
            print("✗ FAIL")
            
        # 生成視覺化
        output_name = f"test_challenging_{test_case['name'].split()[0].lower()}.png"
        detector.visualize_results(test_case['bbox'], results, save_path=output_name)
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    """主測試程式"""
    print("Testing SEM Defect Detector v1 on challenging defect image")
    print("="*60)
    
    # 初始化檢測器
    detector = SEMDefectDetector('challenging_defect_image.jpg')
    
    # 測試每個缺陷
    all_results = []
    for test_case in defect_tests:
        result = test_defect(detector, test_case)
        if result:
            all_results.append((test_case, result))
    
    # 總結
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    passed = 0
    for test_case, result in all_results:
        status = "PASS" if test_case['expected'] in result['defect_type'] else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"{test_case['name']}: {result['defect_type']} - {status}")
    
    print(f"\nTotal: {passed}/{len(all_results)} tests passed")


if __name__ == "__main__":
    main()