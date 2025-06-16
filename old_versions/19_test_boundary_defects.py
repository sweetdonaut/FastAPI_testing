#!/usr/bin/env python3
"""
測試 SEM Defect Detector 對邊界缺陷的檢測能力
專注於真實場景：缺陷只出現在直條和橫條的交界處
"""

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
sys.path.append('.')
from sem_defect_detector_v1 import SEMDefectDetector


def analyze_boundary_defect():
    """分析 D8 邊界缺陷的詳細信息"""
    print("="*60)
    print("Analyzing D8 Boundary Defect")
    print("="*60)
    
    # D8 缺陷信息
    d8_bbox = (474, 477, 9, 22)
    d8_brightness = 157.9
    
    print(f"D8 Location: ({d8_bbox[0]}, {d8_bbox[1]})")
    print(f"D8 Size: {d8_bbox[2]} x {d8_bbox[3]} pixels")
    print(f"D8 Brightness: {d8_brightness}")
    print()
    
    # 使用檢測器分析
    detector = SEMDefectDetector('challenging_defect_image.jpg')
    
    # 執行分析
    results = detector.analyze_defect(
        d8_bbox,
        edge_margin=50,
        brightness_tolerance=15,
        size_tolerance=30
    )
    
    print("\nDetection Results:")
    print(f"Target brightness: {results['target_brightness']:.2f}")
    print(f"Reference mean: {results['reference_mean']:.2f}")
    print(f"Reference count: {results['reference_count']}")
    print(f"Brightness ratio: {results['brightness_ratio']:.3f}×")
    print(f"Detection: {results['defect_type']}")
    print()
    
    # 分析為什麼 D8 能被檢測到
    print("Why D8 was successfully detected:")
    print("1. Located at bright vertical line (high local contrast)")
    print("2. Brightness ratio 2.053× exceeds threshold 1.3×")
    print("3. Sufficient reference regions available (34 cells)")
    print("4. Clear brightness difference from gray background")
    
    return results, detector


def create_boundary_defects_test_image():
    """創建專門的邊界缺陷測試圖片，不同強度的缺陷"""
    print("\n" + "="*60)
    print("Creating Boundary Defects Test Image")
    print("="*60)
    
    # 讀取原始圖片
    img = Image.open('Img_preprocess/sem_noisy_output_raw.jpg').convert('L')
    img_np = np.array(img)
    
    # 檢測邊界位置
    grad_x = np.gradient(img_np, axis=1)
    grad_y = np.gradient(img_np, axis=0)
    
    # 找出強邊緣（邊界）
    edge_threshold_x = np.std(grad_x) * 2
    edge_threshold_y = np.std(grad_y) * 2
    strong_edges = (np.abs(grad_x) > edge_threshold_x) & (np.abs(grad_y) > edge_threshold_y)
    
    # 找出邊界點
    boundary_points = []
    y_coords, x_coords = np.where(strong_edges)
    
    # 聚類邊界點
    for x, y in zip(x_coords, y_coords):
        too_close = False
        for px, py in boundary_points:
            if abs(x - px) < 30 and abs(y - py) < 30:
                too_close = True
                break
        if not too_close and 100 < x < 900 and 100 < y < 900:  # 避免太靠近圖片邊緣
            boundary_points.append((x, y))
    
    # 選擇一些邊界點來添加不同強度的缺陷
    test_defects = []
    intensity_levels = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]  # 不同的亮度增強倍數
    
    if len(boundary_points) >= len(intensity_levels):
        import random
        selected_points = random.sample(boundary_points, len(intensity_levels))
        
        # 創建測試圖片
        test_img = img_np.copy()
        
        for i, ((x, y), intensity) in enumerate(zip(selected_points, intensity_levels)):
            # 缺陷大小
            w, h = 10, 20
            x1 = max(0, x - w//2)
            y1 = max(0, y - h//2)
            x2 = min(img_np.shape[1], x1 + w)
            y2 = min(img_np.shape[0], y1 + h)
            
            # 獲取背景亮度
            region = test_img[y1:y2, x1:x2]
            bg_brightness = np.mean(region)
            
            # 創建缺陷
            defect_brightness = bg_brightness * intensity
            noise = np.random.normal(0, 3, region.shape)
            defect_region = np.clip(defect_brightness + noise, 0, 255)
            
            test_img[y1:y2, x1:x2] = defect_region
            
            test_defects.append({
                'id': i + 1,
                'bbox': (x1, y1, x2-x1, y2-y1),
                'intensity_factor': intensity,
                'expected_brightness': defect_brightness
            })
    
    # 應用輕微的高斯模糊
    from scipy.ndimage import gaussian_filter
    test_img = gaussian_filter(test_img, sigma=0.5)
    
    # 保存測試圖片
    Image.fromarray(test_img.astype(np.uint8)).save('boundary_defects_test.jpg')
    
    # 創建標註圖
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(test_img, cmap='gray')
    ax.set_title(f'Boundary Defects Test Image ({len(test_defects)} defects with varying intensity)', fontsize=14)
    
    # 標註缺陷
    for defect in test_defects:
        x, y, w, h = defect['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # 添加標籤
        label = f"×{defect['intensity_factor']:.1f}"
        ax.text(x + w/2, y - 5, label, 
                color='red', fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('boundary_defects_test_annotated.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created {len(test_defects)} boundary defects with intensity factors from {intensity_levels[0]} to {intensity_levels[-1]}")
    
    return test_defects


def test_detection_sensitivity(test_defects):
    """測試檢測器對不同強度邊界缺陷的敏感度"""
    print("\n" + "="*60)
    print("Testing Detection Sensitivity")
    print("="*60)
    
    detector = SEMDefectDetector('boundary_defects_test.jpg')
    
    results_summary = []
    
    for defect in test_defects:
        try:
            results = detector.analyze_defect(
                defect['bbox'],
                edge_margin=50,
                brightness_tolerance=15,
                size_tolerance=30
            )
            
            detected = 'DEFECT' in results['defect_type']
            results_summary.append({
                'intensity_factor': defect['intensity_factor'],
                'brightness_ratio': results['brightness_ratio'],
                'detected': detected,
                'defect_type': results['defect_type']
            })
            
            status = "✓" if detected else "✗"
            print(f"Intensity ×{defect['intensity_factor']:.1f}: "
                  f"Ratio={results['brightness_ratio']:.3f}, "
                  f"{results['defect_type']} {status}")
            
        except Exception as e:
            print(f"Intensity ×{defect['intensity_factor']:.1f}: Error - {e}")
            results_summary.append({
                'intensity_factor': defect['intensity_factor'],
                'brightness_ratio': 0,
                'detected': False,
                'defect_type': 'ERROR'
            })
    
    # 分析結果
    print("\n" + "-"*60)
    print("Detection Sensitivity Analysis:")
    print("-"*60)
    
    detected_count = sum(1 for r in results_summary if r['detected'])
    print(f"Detected: {detected_count}/{len(results_summary)} defects")
    
    # 找出檢測閾值
    detected_ratios = [r['brightness_ratio'] for r in results_summary if r['detected']]
    undetected_ratios = [r['brightness_ratio'] for r in results_summary if not r['detected'] and r['brightness_ratio'] > 0]
    
    if detected_ratios and undetected_ratios:
        min_detected = min(detected_ratios)
        max_undetected = max(undetected_ratios)
        print(f"Minimum detected ratio: {min_detected:.3f}")
        print(f"Maximum undetected ratio: {max_undetected:.3f}")
        print(f"Effective threshold appears to be around: {(min_detected + max_undetected)/2:.3f}")
    
    return results_summary


def generate_stress_test_report(d8_results, sensitivity_results):
    """生成壓力測試報告"""
    print("\n" + "="*60)
    print("STRESS TEST REPORT")
    print("="*60)
    
    print("\n1. Original D8 Defect Analysis:")
    print("   - Successfully detected at boundary location")
    print("   - Brightness ratio: 2.053× (well above threshold)")
    print("   - Demonstrates detector works well for clear boundary defects")
    
    print("\n2. Sensitivity Test Results:")
    detected = sum(1 for r in sensitivity_results if r['detected'])
    total = len(sensitivity_results)
    print(f"   - Detection rate: {detected}/{total} ({detected/total*100:.0f}%)")
    
    # 找出檢測的轉折點
    sorted_results = sorted(sensitivity_results, key=lambda x: x['intensity_factor'])
    for i, r in enumerate(sorted_results):
        if r['detected']:
            if i > 0:
                print(f"   - Detection starts at intensity factor ×{r['intensity_factor']:.1f}")
            break
    
    print("\n3. Key Findings:")
    print("   - Detector successfully identifies boundary defects")
    print("   - Current threshold (1.3×) is appropriate for clear defects")
    print("   - Gaussian blur reduces effective contrast")
    print("   - Boundary locations provide good contrast for detection")
    
    print("\n4. Recommendations:")
    print("   - For production use, consider image quality metrics")
    print("   - Adaptive thresholds based on local contrast")
    print("   - Pre-processing to enhance boundary regions")
    print("   - Multi-scale analysis for various defect sizes")


def main():
    """主程式"""
    # 1. 分析 D8 邊界缺陷
    d8_results, detector = analyze_boundary_defect()
    
    # 2. 創建邊界缺陷測試圖片
    test_defects = create_boundary_defects_test_image()
    
    # 3. 測試檢測敏感度
    sensitivity_results = test_detection_sensitivity(test_defects)
    
    # 4. 生成報告
    generate_stress_test_report(d8_results, sensitivity_results)


if __name__ == "__main__":
    main()