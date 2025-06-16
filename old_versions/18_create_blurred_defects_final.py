#!/usr/bin/env python3
"""
基於實際圖片分析創建模糊邊緣缺陷
使用檢測到的實際亮條位置
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter, sobel
from scipy.signal import find_peaks
import os


def analyze_and_detect_structure(image_np):
    """分析圖片結構，找出垂直邊緣和水平亮條"""
    print("Analyzing image structure...")
    
    # 1. 檢測垂直邊緣（使用Sobel）
    edges_v = sobel(image_np, axis=1)
    edge_strength = np.mean(np.abs(edges_v), axis=0)
    
    # 找垂直邊緣
    v_threshold = np.percentile(edge_strength, 85)
    v_peaks, _ = find_peaks(edge_strength, height=v_threshold, distance=40)
    
    # 驗證邊緣
    vertical_edges = []
    for peak in v_peaks:
        if 20 < peak < image_np.shape[1] - 20:
            left_val = np.mean(image_np[:, max(0, peak-20):peak-5])
            right_val = np.mean(image_np[:, peak+5:min(image_np.shape[1], peak+20)])
            if abs(left_val - right_val) > 20:
                vertical_edges.append(peak)
    
    print(f"Found {len(vertical_edges)} vertical edges")
    
    # 2. 檢測水平亮條（基於行亮度分析）
    row_means = np.mean(image_np, axis=1)
    
    # 找亮度峰值 - 使用更寬鬆的參數
    # 基於分析結果，亮條在 y=54, 290, 363, 488, 610, 734, 856
    # 使用相對高度而非絕對高度
    mean_brightness = np.mean(row_means)
    std_brightness = np.std(row_means)
    h_peaks, properties = find_peaks(row_means, 
                                    height=mean_brightness + 0.5 * std_brightness,
                                    distance=80,
                                    prominence=3)
    
    # 為每個峰值創建條帶
    horizontal_bands = []
    for peak in h_peaks:
        # 找峰值周圍的亮區域
        start = peak
        end = peak
        
        threshold = row_means[peak] - 3  # 峰值減3
        
        # 向上擴展
        while start > 0 and row_means[start] > threshold:
            start -= 1
        
        # 向下擴展
        while end < len(row_means) - 1 and row_means[end] > threshold:
            end += 1
        
        if 15 <= end - start <= 100:  # 合理的寬度範圍
            horizontal_bands.append((start, end))
    
    print(f"Found {len(horizontal_bands)} horizontal bright bands")
    
    return vertical_edges, horizontal_bands


def create_final_blurred_defects():
    """創建最終的模糊邊緣缺陷圖片"""
    # 讀取原始圖片
    input_path = 'Img_preprocess/sem_noisy_output_raw.jpg'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        return
    
    img = Image.open(input_path).convert('L')
    img_np = np.array(img)
    
    # 分析原始圖片結構
    vertical_edges, horizontal_bands = analyze_and_detect_structure(img_np)
    
    # 創建模糊基礎圖片
    print("Creating blurred base image...")
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
                # 在模糊圖上驗證仍有對比度
                left_region = blurred_base[band_center-10:band_center+10, v_edge-20:v_edge-5]
                right_region = blurred_base[band_center-10:band_center+10, v_edge+5:v_edge+20]
                
                if left_region.size > 0 and right_region.size > 0:
                    contrast = abs(np.mean(left_region) - np.mean(right_region))
                    
                    if contrast > 5:  # 降低模糊後的對比度閾值
                        valid_locations.append({
                            'x': v_edge,
                            'y': band_center,
                            'contrast': contrast,
                            'band_start': band_start,
                            'band_end': band_end
                        })
    
    # 去重
    filtered_locations = []
    for loc in valid_locations:
        too_close = False
        for existing in filtered_locations:
            if abs(loc['x'] - existing['x']) < 100 and abs(loc['y'] - existing['y']) < 100:
                too_close = True
                break
        if not too_close:
            filtered_locations.append(loc)
    
    filtered_locations.sort(key=lambda x: x['y'])
    print(f"Found {len(filtered_locations)} valid defect locations")
    
    if len(filtered_locations) == 0:
        print("No valid locations found!")
        return
    
    # 添加缺陷
    defect_img = blurred_base.copy()
    defect_info = []
    
    # 選擇缺陷
    num_defects = min(10, len(filtered_locations))
    if num_defects < len(filtered_locations):
        # 均勻分布
        indices = np.linspace(0, len(filtered_locations)-1, num_defects, dtype=int)
        selected_locations = [filtered_locations[i] for i in indices]
    else:
        selected_locations = filtered_locations
    
    # 缺陷強度
    intensity_factors = np.linspace(1.3, 2.2, len(selected_locations))
    
    for i, (loc, intensity) in enumerate(zip(selected_locations, intensity_factors)):
        x, y = loc['x'], loc['y']
        
        # 橢圓形缺陷
        w = 10 + np.random.randint(-2, 3)
        h = int(w * 1.5)  # 垂直方向更長
        
        x1 = max(0, x - w//2)
        y1 = max(0, y - h//2)
        x2 = min(defect_img.shape[1], x1 + w)
        y2 = min(defect_img.shape[0], y1 + h)
        
        # 創建橢圓缺陷
        region = defect_img[y1:y2, x1:x2]
        if region.shape[0] > 2 and region.shape[1] > 2:
            h, w = region.shape
            y_grid, x_grid = np.ogrid[:h, :w]
            cy, cx = h/2, w/2
            
            # 橢圓遮罩
            a = max(1, w/2 * 0.8)
            b = max(1, h/2 * 0.9)
            distance = np.sqrt(((x_grid - cx)/a)**2 + ((y_grid - cy)/b)**2)
            mask = distance <= 1
            gradient = np.exp(-2 * distance) * 0.7
            
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
                'band': (loc['band_start'], loc['band_end'])
            })
    
    # 最終輕微模糊
    final_img = gaussian_filter(defect_img, sigma=0.3)
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)
    
    # 保存
    Image.fromarray(final_img).save('blurred_defects_final.jpg')
    
    # 視覺化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
    
    # 原始圖片
    ax1.imshow(img_np, cmap='gray')
    ax1.set_title('Original Image', fontsize=14)
    ax1.axis('off')
    
    # 結構檢測
    ax2.imshow(img_np, cmap='gray')
    ax2.set_title('Detected Structure', fontsize=14)
    
    # 垂直邊緣
    for v_edge in vertical_edges:
        ax2.axvline(x=v_edge, color='red', linewidth=2, alpha=0.8)
    
    # 水平亮條
    for band_start, band_end in horizontal_bands:
        rect = patches.Rectangle((0, band_start), img_np.shape[1], band_end-band_start,
                               linewidth=0, facecolor='yellow', alpha=0.3)
        ax2.add_patch(rect)
        # 標記中心線
        center = (band_start + band_end) // 2
        ax2.axhline(y=center, color='yellow', linewidth=1, alpha=0.8, linestyle='--')
    
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
    ax3.set_title('Blurred Base (sigma=2.5)', fontsize=14)
    ax3.axis('off')
    
    # 最終圖
    ax4.imshow(final_img, cmap='gray')
    ax4.set_title(f'Final: {len(defect_info)} Defects', fontsize=14)
    
    # 標註缺陷
    for defect in defect_info:
        cx, cy = defect['center']
        w, h = defect['bbox'][2], defect['bbox'][3]
        
        ellipse = patches.Ellipse((cx, cy), w, h, linewidth=2,
                                edgecolor='red', facecolor='none')
        ax4.add_patch(ellipse)
        
        label = f"D{defect['id']}\n×{defect['intensity_factor']:.1f}"
        ax4.text(cx, cy - h/2 - 15, label, 
                color='red', fontsize=9, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('blurred_defects_final_annotated.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 打印信息
    print("\nDefect Information:")
    print("=" * 80)
    print(f"{'ID':<4} {'Center':<12} {'Size':<10} {'Intensity':<10} {'Band Y':<15} {'Contrast':<10}")
    print("-" * 80)
    
    for defect in defect_info:
        cx, cy = defect['center']
        w, h = defect['bbox'][2], defect['bbox'][3]
        band_start, band_end = defect['band']
        
        print(f"D{defect['id']:<3} ({cx:>3},{cy:>3})     "
              f"{w:>2}×{h:<3}      ×{defect['intensity_factor']:<9.1f} "
              f"y={band_start:>3}-{band_end:<3}     {defect['contrast']:>6.1f}")
    
    # 保存信息
    with open('blurred_defects_final_info.txt', 'w') as f:
        f.write("Final Blurred Edge Defects\n")
        f.write("=" * 80 + "\n")
        f.write("- Based on actual bright band detection\n")
        f.write("- Vertical edges properly aligned\n")
        f.write("- Heavy blur (sigma=2.5) applied\n")
        f.write(f"- Edge margin: {edge_margin}px\n\n")
        
        for defect in defect_info:
            f.write(f"\nDefect {defect['id']}:\n")
            f.write(f"  Bbox: {defect['bbox']}\n")
            f.write(f"  Center: {defect['center']}\n")
            f.write(f"  Intensity: ×{defect['intensity_factor']:.1f}\n")
            f.write(f"  Band: y={defect['band'][0]}-{defect['band'][1]}\n")
            f.write(f"  Contrast: {defect['contrast']:.1f}\n")
    
    print("\nFiles created:")
    print("- blurred_defects_final.jpg")
    print("- blurred_defects_final_annotated.png")
    print("- blurred_defects_final_info.txt")


if __name__ == "__main__":
    create_final_blurred_defects()