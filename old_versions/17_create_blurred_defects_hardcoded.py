#!/usr/bin/env python3
"""
使用硬編碼的亮條位置創建模糊邊緣缺陷
基於 brightness_analysis.py 的結果
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter, sobel
from scipy.signal import find_peaks
import os


def get_vertical_edges(image_np):
    """檢測垂直邊緣"""
    edges_v = sobel(image_np, axis=1)
    edge_strength = np.mean(np.abs(edges_v), axis=0)
    
    v_threshold = np.percentile(edge_strength, 85)
    v_peaks, _ = find_peaks(edge_strength, height=v_threshold, distance=40)
    
    vertical_edges = []
    for peak in v_peaks:
        if 20 < peak < image_np.shape[1] - 20:
            left_val = np.mean(image_np[:, max(0, peak-20):peak-5])
            right_val = np.mean(image_np[:, peak+5:min(image_np.shape[1], peak+20)])
            if abs(left_val - right_val) > 20:
                vertical_edges.append(peak)
    
    return vertical_edges


def create_hardcoded_blurred_defects():
    """使用已知的亮條位置創建缺陷"""
    # 讀取原始圖片
    input_path = 'Img_preprocess/sem_noisy_output_raw.jpg'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        return
    
    img = Image.open(input_path).convert('L')
    img_np = np.array(img)
    
    # 檢測垂直邊緣
    print("Detecting vertical edges...")
    vertical_edges = get_vertical_edges(img_np)
    print(f"Found {len(vertical_edges)} vertical edges")
    
    # 硬編碼的水平亮條（基於分析結果）
    # 峰值在 y=54, 290, 363, 488, 610, 734, 856
    horizontal_bands = [
        (40, 70),    # 圍繞 y=54
        (275, 305),  # 圍繞 y=290
        (350, 380),  # 圍繞 y=363
        (475, 505),  # 圍繞 y=488
        (595, 625),  # 圍繞 y=610
        (720, 750),  # 圍繞 y=734
        (840, 870),  # 圍繞 y=856
    ]
    print(f"Using {len(horizontal_bands)} predefined horizontal bands")
    
    # 創建模糊基礎圖片
    print("Creating heavily blurred base image...")
    blurred_base = gaussian_filter(img_np.astype(float), sigma=2.5)
    noise = np.random.normal(0, 1, blurred_base.shape)
    blurred_base = np.clip(blurred_base + noise, 0, 255)
    
    # 找出有效的缺陷位置
    edge_margin = 100
    valid_locations = []
    
    for v_edge in vertical_edges:
        if v_edge < edge_margin or v_edge > img_np.shape[1] - edge_margin:
            continue
        
        for band_start, band_end in horizontal_bands:
            band_center = (band_start + band_end) // 2
            
            if edge_margin < band_center < img_np.shape[0] - edge_margin:
                # 計算模糊圖上的對比度
                y1 = max(0, band_center - 15)
                y2 = min(blurred_base.shape[0], band_center + 15)
                x1_left = max(0, v_edge - 25)
                x2_left = max(0, v_edge - 5)
                x1_right = min(blurred_base.shape[1], v_edge + 5)
                x2_right = min(blurred_base.shape[1], v_edge + 25)
                
                left_region = blurred_base[y1:y2, x1_left:x2_left]
                right_region = blurred_base[y1:y2, x1_right:x2_right]
                
                if left_region.size > 0 and right_region.size > 0:
                    contrast = abs(np.mean(left_region) - np.mean(right_region))
                    
                    # 即使對比度很低也接受（因為模糊很強）
                    if contrast > 3:
                        valid_locations.append({
                            'x': v_edge,
                            'y': band_center,
                            'contrast': contrast,
                            'band': (band_start, band_end)
                        })
    
    # 去重
    filtered_locations = []
    for loc in valid_locations:
        too_close = False
        for existing in filtered_locations:
            if abs(loc['x'] - existing['x']) < 80 and abs(loc['y'] - existing['y']) < 80:
                too_close = True
                break
        if not too_close:
            filtered_locations.append(loc)
    
    filtered_locations.sort(key=lambda x: x['y'])
    print(f"Found {len(filtered_locations)} valid defect locations")
    
    if len(filtered_locations) == 0:
        print("No valid locations found! Check edge margins or contrast threshold.")
        return
    
    # 添加缺陷
    defect_img = blurred_base.copy()
    defect_info = []
    
    # 選擇缺陷位置（最多10個）
    num_defects = min(10, len(filtered_locations))
    if num_defects < len(filtered_locations):
        indices = np.linspace(0, len(filtered_locations)-1, num_defects, dtype=int)
        selected_locations = [filtered_locations[i] for i in indices]
    else:
        selected_locations = filtered_locations
    
    # 缺陷強度（需要更高因為背景很模糊）
    intensity_factors = np.linspace(1.4, 2.5, len(selected_locations))
    
    for i, (loc, intensity) in enumerate(zip(selected_locations, intensity_factors)):
        x, y = loc['x'], loc['y']
        
        # 橢圓形缺陷
        w = 12 + np.random.randint(-2, 3)
        h = int(w * 1.5)
        
        x1 = max(0, x - w//2)
        y1 = max(0, y - h//2)
        x2 = min(defect_img.shape[1], x1 + w)
        y2 = min(defect_img.shape[0], y1 + h)
        
        # 創建橢圓缺陷
        region = defect_img[y1:y2, x1:x2]
        if region.shape[0] > 3 and region.shape[1] > 3:
            h_r, w_r = region.shape
            y_grid, x_grid = np.ogrid[:h_r, :w_r]
            cy, cx = h_r/2, w_r/2
            
            # 橢圓
            a = max(1, w_r/2 * 0.8)
            b = max(1, h_r/2 * 0.9)
            distance = np.sqrt(((x_grid - cx)/a)**2 + ((y_grid - cy)/b)**2)
            gradient = np.exp(-1.5 * distance) * 0.8
            
            # 應用缺陷
            base_brightness = np.mean(region)
            defect_brightness = base_brightness * intensity
            noise = np.random.normal(0, 2, region.shape)
            
            region_new = region * (1 - gradient) + (defect_brightness + noise) * gradient
            defect_img[y1:y2, x1:x2] = np.clip(region_new, 0, 255)
            
            defect_info.append({
                'id': i + 1,
                'bbox': (x1, y1, x2-x1, y2-y1),
                'center': (x, y),
                'intensity_factor': intensity,
                'contrast': loc['contrast'],
                'band': loc['band']
            })
    
    # 最終輕微模糊
    final_img = gaussian_filter(defect_img, sigma=0.2)
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)
    
    # 保存
    Image.fromarray(final_img).save('blurred_defects_hardcoded.jpg')
    
    # 視覺化
    fig = plt.figure(figsize=(20, 16))
    
    # 創建網格
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])
    
    # 原始圖片
    ax1.imshow(img_np, cmap='gray')
    ax1.set_title('Original Image', fontsize=14)
    ax1.axis('off')
    
    # 結構（在原圖上顯示）
    ax2.imshow(img_np, cmap='gray')
    ax2.set_title('Structure (on Original)', fontsize=14)
    
    # 垂直邊緣
    for v_edge in vertical_edges:
        ax2.axvline(x=v_edge, color='red', linewidth=2, alpha=0.8)
    
    # 水平亮條
    for band_start, band_end in horizontal_bands:
        rect = patches.Rectangle((0, band_start), img_np.shape[1], band_end-band_start,
                               linewidth=0, facecolor='yellow', alpha=0.3)
        ax2.add_patch(rect)
    
    # 邊界
    margin_rect = patches.Rectangle((edge_margin, edge_margin), 
                                   img_np.shape[1]-2*edge_margin, 
                                   img_np.shape[0]-2*edge_margin,
                                   linewidth=2, edgecolor='blue', 
                                   facecolor='none', linestyle='--')
    ax2.add_patch(margin_rect)
    
    ax2.axis('off')
    
    # 模糊圖
    ax3.imshow(blurred_base, cmap='gray', vmin=0, vmax=255)
    ax3.set_title('Heavily Blurred Base (sigma=2.5)', fontsize=14)
    ax3.axis('off')
    
    # 最終圖（大圖）
    ax4.imshow(final_img, cmap='gray')
    ax4.set_title(f'Final Result: {len(defect_info)} Defects on Heavily Blurred Edges', fontsize=16)
    
    # 標註缺陷
    for defect in defect_info:
        cx, cy = defect['center']
        w, h = defect['bbox'][2], defect['bbox'][3]
        
        ellipse = patches.Ellipse((cx, cy), w, h, linewidth=2.5,
                                edgecolor='red', facecolor='none')
        ax4.add_patch(ellipse)
        
        label = f"D{defect['id']}\n×{defect['intensity_factor']:.1f}"
        ax4.text(cx, cy - h/2 - 20, label, 
                color='red', fontsize=11, ha='center', weight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.9))
    
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('blurred_defects_hardcoded_annotated.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 打印結果
    print("\n" + "="*80)
    print("DEFECT CREATION SUMMARY")
    print("="*80)
    print(f"Successfully created {len(defect_info)} defects on heavily blurred edges")
    print("\nDefect locations:")
    for defect in defect_info:
        cx, cy = defect['center']
        print(f"  D{defect['id']}: ({cx}, {cy}) - Intensity ×{defect['intensity_factor']:.1f}, "
              f"Contrast {defect['contrast']:.1f}")
    
    print("\nChallenges in this test image:")
    print("- Heavy Gaussian blur (sigma=2.5) makes edges very soft")
    print("- Low contrast between regions after blurring")
    print("- Defects need higher intensity factors (1.4-2.5×)")
    print("- Detection will be challenging for standard algorithms")
    
    # 保存信息
    with open('blurred_defects_hardcoded_info.txt', 'w') as f:
        f.write("Heavily Blurred Edge Defects\n")
        f.write("="*80 + "\n")
        f.write("Test image with extreme blur to challenge detection algorithms\n\n")
        f.write("Image properties:\n")
        f.write("- Base blur: sigma=2.5 (heavy)\n")
        f.write("- Final blur: sigma=0.2 (light)\n")
        f.write(f"- Edge margin: {edge_margin}px\n")
        f.write(f"- Defect intensity range: {intensity_factors[0]:.1f}-{intensity_factors[-1]:.1f}×\n\n")
        
        for defect in defect_info:
            f.write(f"\nDefect {defect['id']}:\n")
            f.write(f"  Bbox: {defect['bbox']}\n")
            f.write(f"  Center: {defect['center']}\n")
            f.write(f"  Intensity: ×{defect['intensity_factor']:.1f}\n")
            f.write(f"  Local contrast: {defect['contrast']:.1f}\n")
            f.write(f"  Band: y={defect['band'][0]}-{defect['band'][1]}\n")
    
    print("\nFiles created:")
    print("- blurred_defects_hardcoded.jpg")
    print("- blurred_defects_hardcoded_annotated.png")
    print("- blurred_defects_hardcoded_info.txt")


if __name__ == "__main__":
    create_hardcoded_blurred_defects()