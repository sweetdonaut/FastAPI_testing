#!/usr/bin/env python3
"""
測試 SEM 缺陷檢測器 v2 對模糊邊緣圖片的效果
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

# 導入檢測器 v2
from sem_defect_detector_v2 import SEMDefectDetector


def read_defect_info():
    """讀取缺陷資訊"""
    defects = []
    
    with open('final_blurred_defects_info.txt', 'r') as f:
        lines = f.readlines()
    
    current_defect = None
    for line in lines:
        line = line.strip()
        if line.startswith('Defect '):
            if current_defect:
                defects.append(current_defect)
            current_defect = {'id': int(line.split()[1].rstrip(':'))}
        elif current_defect and line.startswith('Bbox:'):
            # 解析 bbox: (np.int64(121), 281, np.int64(12), 19)
            bbox_str = line.split(':', 1)[1].strip()
            # 移除 np.int64() 包裝
            bbox_str = bbox_str.replace('np.int64(', '').replace(')', '')
            parts = [int(x.strip()) for x in bbox_str.strip('()').split(',')]
            current_defect['bbox'] = tuple(parts)  # (x, y, w, h)
        elif current_defect and line.startswith('Intensity:'):
            intensity_str = line.split('×')[1].strip()
            current_defect['intensity'] = float(intensity_str)
    
    if current_defect:
        defects.append(current_defect)
    
    return defects


def test_single_defect(detector, defect_info, save_prefix='test_defect_v2'):
    """測試單個缺陷"""
    defect_id = defect_info['id']
    target_bbox = defect_info['bbox']
    expected_intensity = defect_info['intensity']
    
    print(f"\n{'='*60}")
    print(f"Testing Defect {defect_id}")
    print(f"Bbox: {target_bbox}")
    print(f"Expected intensity: {expected_intensity}×")
    print('='*60)
    
    try:
        # 執行分析
        results = detector.analyze_defect(target_bbox, 
                                        edge_margin=100,
                                        brightness_tolerance=10, 
                                        size_tolerance=20)
        
        # 打印結果
        print(f"\nTarget brightness: {results['target_brightness']:.2f}")
        print(f"Reference mean: {results['reference_mean']:.2f}")
        print(f"Reference count: {results['reference_count']}")
        print(f"Brightness ratio: {results['brightness_ratio']:.3f}×")
        print(f"Detection result: {results['defect_type']}")
        
        # 判斷是否成功檢測
        detected = results['defect_type'] == 'BRIGHT DEFECT'
        success = detected and results['brightness_ratio'] > 1.3
        
        print(f"\nDetection {'SUCCESS' if success else 'FAILED'}")
        
        # 生成個別視覺化
        save_path = f"{save_prefix}_{defect_id:02d}.png"
        detector.visualize_results(target_bbox, results, save_path=save_path)
        
        return {
            'defect_id': defect_id,
            'success': success,
            'detected': detected,
            'brightness_ratio': results['brightness_ratio'],
            'reference_count': results['reference_count'],
            'results': results
        }
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return {
            'defect_id': defect_id,
            'success': False,
            'detected': False,
            'error': str(e)
        }


def create_summary_visualization(test_results, image_path='final_blurred_defects.jpg'):
    """創建總結視覺化"""
    img = Image.open(image_path)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(img, cmap='gray')
    plt.title('Blurred Defects Detection Summary (v2)', fontsize=16, weight='bold')
    
    # 繪製所有缺陷的檢測結果
    for result in test_results:
        if 'error' not in result:
            defect_info = next(d for d in defects if d['id'] == result['defect_id'])
            x, y, w, h = defect_info['bbox']
            
            # 根據檢測結果選擇顏色
            if result['success']:
                color = 'green'
                label = f"D{result['defect_id']}: ✓"
            elif result['detected']:
                color = 'yellow'
                label = f"D{result['defect_id']}: {result['brightness_ratio']:.2f}×"
            else:
                color = 'red'
                label = f"D{result['defect_id']}: ✗"
            
            # 繪製邊框
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)
            
            # 添加標籤
            plt.text(x + w/2, y - 5, label, color=color, fontsize=8, 
                    ha='center', va='bottom', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 添加圖例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
               markersize=10, label='Detection Success'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', 
               markersize=10, label='Weak Detection'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
               markersize=10, label='Detection Failed')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # 添加統計資訊
    success_count = sum(1 for r in test_results if r.get('success', False))
    detected_count = sum(1 for r in test_results if r.get('detected', False))
    total_count = len(test_results)
    
    stats_text = f"Detection Stats (v2):\n"
    stats_text += f"Total defects: {total_count}\n"
    stats_text += f"Successfully detected: {success_count} ({success_count/total_count*100:.1f}%)\n"
    stats_text += f"Weakly detected: {detected_count - success_count}\n"
    stats_text += f"Failed: {total_count - detected_count}"
    
    plt.text(20, 50, stats_text, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('blurred_detection_summary_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSummary visualization saved as 'blurred_detection_summary_v2.png'")


if __name__ == "__main__":
    # 讀取缺陷資訊
    print("Reading defect information...")
    defects = read_defect_info()
    print(f"Found {len(defects)} defects")
    
    # 初始化檢測器
    print("\nInitializing detector v2 with blurred image...")
    detector = SEMDefectDetector('final_blurred_defects.jpg')
    
    # 測試所有缺陷
    test_results = []
    for defect in defects:
        result = test_single_defect(detector, defect, save_prefix='blurred_test_v2')
        test_results.append(result)
    
    # 打印總結
    print("\n" + "="*80)
    print("FINAL SUMMARY (v2)")
    print("="*80)
    
    success_count = sum(1 for r in test_results if r.get('success', False))
    detected_count = sum(1 for r in test_results if r.get('detected', False))
    failed_count = sum(1 for r in test_results if 'error' in r)
    
    print(f"Total defects tested: {len(defects)}")
    print(f"Successfully detected (ratio > 1.3): {success_count}")
    print(f"Weakly detected (identified as bright but ratio < 1.3): {detected_count - success_count}")
    print(f"Not detected: {len(defects) - detected_count - failed_count}")
    print(f"Errors: {failed_count}")
    
    # 比較改進
    print("\nImprovement from v1:")
    print("v1: 8/12 (66.7%) success rate")
    print(f"v2: {success_count}/12 ({success_count/12*100:.1f}%) success rate")
    
    if failed_count > 0:
        print("\nErrors encountered:")
        for r in test_results:
            if 'error' in r:
                print(f"  Defect {r['defect_id']}: {r['error']}")
    
    # 創建總結視覺化
    create_summary_visualization(test_results)
    
    print("\nTest completed!")