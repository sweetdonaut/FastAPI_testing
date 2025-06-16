#!/usr/bin/env python3
"""
創建最終的模糊邊緣缺陷圖片
使用正確的垂直邊緣位置
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import os


def get_accurate_vertical_edges(img_np):
    """準確檢測垂直邊緣"""
    # 計算列平均亮度
    column_means = np.mean(img_np, axis=0)
    
    # 計算梯度
    gradient = np.gradient(column_means)
    
    # 找正梯度峰值（從暗到亮的邊緣）
    pos_peaks, _ = find_peaks(gradient, height=np.std(gradient)*1.5, distance=30)
    
    # 找負梯度峰值（從亮到暗的邊緣）
    neg_peaks, _ = find_peaks(-gradient, height=np.std(gradient)*1.5, distance=30)
    
    # 合併所有邊緣
    all_edges = sorted(list(pos_peaks) + list(neg_peaks))
    
    # 標記邊緣類型
    edge_info = []
    for edge in all_edges:
        if edge in pos_peaks:
            edge_info.append({'pos': edge, 'type': 'dark_to_bright'})
        else:
            edge_info.append({'pos': edge, 'type': 'bright_to_dark'})
    
    return edge_info


def create_final_realistic_defects():
    """創建最終的真實缺陷圖片"""
    # 讀取原始圖片
    input_path = 'Img_preprocess/sem_noisy_output_raw.jpg'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        return
    
    img = Image.open(input_path).convert('L')
    img_np = np.array(img)
    
    print("Detecting accurate vertical edges...")
    edge_info = get_accurate_vertical_edges(img_np)
    print(f"Found {len(edge_info)} vertical edges")
    
    # 使用之前分析得到的水平亮條位置
    horizontal_bands = [
        (40, 70),    # y=54
        (275, 305),  # y=290
        (350, 380),  # y=363
        (475, 505),  # y=488
        (595, 625),  # y=610
        (720, 750),  # y=734
        (840, 870),  # y=856
    ]
    
    # 創建強模糊的基礎圖片
    print("Creating heavily blurred base image (sigma=2.5)...")
    blurred_base = gaussian_filter(img_np.astype(float), sigma=2.5)
    noise = np.random.normal(0, 1, blurred_base.shape)
    blurred_base = np.clip(blurred_base + noise, 0, 255)
    
    # 找出有效的缺陷位置
    edge_margin = 100
    valid_locations = []
    
    for edge in edge_info:
        edge_x = edge['pos']
        edge_type = edge['type']
        
        # 跳過太靠近邊界的邊緣
        if edge_x < edge_margin or edge_x > img_np.shape[1] - edge_margin:
            continue
        
        for band_start, band_end in horizontal_bands:
            band_center = (band_start + band_end) // 2
            
            if edge_margin < band_center < img_np.shape[0] - edge_margin:
                # 驗證這確實是一個好的缺陷位置
                # 在模糊圖上檢查對比度
                y1 = max(0, band_center - 15)
                y2 = min(blurred_base.shape[0], band_center + 15)
                
                # 根據邊緣類型決定左右區域
                if edge_type == 'dark_to_bright':
                    # 左邊暗，右邊亮
                    left_region = blurred_base[y1:y2, max(0, edge_x-25):max(0, edge_x-5)]
                    right_region = blurred_base[y1:y2, min(blurred_base.shape[1], edge_x+5):min(blurred_base.shape[1], edge_x+25)]
                else:
                    # 左邊亮，右邊暗
                    left_region = blurred_base[y1:y2, max(0, edge_x-25):max(0, edge_x-5)]
                    right_region = blurred_base[y1:y2, min(blurred_base.shape[1], edge_x+5):min(blurred_base.shape[1], edge_x+25)]
                
                if left_region.size > 0 and right_region.size > 0:
                    contrast = abs(np.mean(left_region) - np.mean(right_region))
                    
                    # 即使對比度較低也接受（因為模糊）
                    if contrast > 3:
                        valid_locations.append({
                            'x': edge_x,
                            'y': band_center,
                            'contrast': contrast,
                            'edge_type': edge_type,
                            'band': (band_start, band_end)
                        })
    
    # 去重（避免在同一位置放多個缺陷）
    filtered_locations = []
    for loc in valid_locations:
        too_close = False
        for existing in filtered_locations:
            if abs(loc['x'] - existing['x']) < 60 and abs(loc['y'] - existing['y']) < 60:
                too_close = True
                break
        if not too_close:
            filtered_locations.append(loc)
    
    # 按y坐標排序
    filtered_locations.sort(key=lambda x: x['y'])
    print(f"Found {len(filtered_locations)} valid defect locations")
    
    # 創建缺陷
    defect_img = blurred_base.copy()
    defect_info = []
    
    # 選擇缺陷（最多12個，確保覆蓋不同區域）
    num_defects = min(12, len(filtered_locations))
    if num_defects < len(filtered_locations):
        # 均勻選擇
        indices = np.linspace(0, len(filtered_locations)-1, num_defects, dtype=int)
        selected_locations = [filtered_locations[i] for i in indices]
    else:
        selected_locations = filtered_locations
    
    # 缺陷強度（因為背景模糊，需要更高強度）
    intensity_factors = np.linspace(1.4, 2.5, len(selected_locations))
    
    for i, (loc, intensity) in enumerate(zip(selected_locations, intensity_factors)):
        x, y = loc['x'], loc['y']
        
        # 橢圓形缺陷（垂直方向稍長）
        w = 10 + np.random.randint(-2, 3)
        h = int(w * 1.6)
        
        x1 = max(0, x - w//2)
        y1 = max(0, y - h//2)
        x2 = min(defect_img.shape[1], x1 + w)
        y2 = min(defect_img.shape[0], y1 + h)
        
        # 創建橢圓形缺陷
        region = defect_img[y1:y2, x1:x2]
        if region.shape[0] > 3 and region.shape[1] > 3:
            h_r, w_r = region.shape
            y_grid, x_grid = np.ogrid[:h_r, :w_r]
            cy, cx = h_r/2, w_r/2
            
            # 橢圓參數
            a = max(1, w_r/2 * 0.8)
            b = max(1, h_r/2 * 0.9)
            
            # 創建柔和的橢圓
            distance = np.sqrt(((x_grid - cx)/a)**2 + ((y_grid - cy)/b)**2)
            gradient = np.exp(-1.5 * distance) * 0.8
            
            # 應用缺陷
            base_brightness = np.mean(region)
            defect_brightness = base_brightness * intensity
            noise = np.random.normal(0, 2, region.shape)
            
            # 混合
            region_new = region * (1 - gradient) + (defect_brightness + noise) * gradient
            defect_img[y1:y2, x1:x2] = np.clip(region_new, 0, 255)
            
            defect_info.append({
                'id': i + 1,
                'bbox': (x1, y1, x2-x1, y2-y1),
                'center': (x, y),
                'intensity_factor': intensity,
                'contrast': loc['contrast'],
                'edge_type': loc['edge_type'],
                'band': loc['band']
            })
    
    # 最終輕微模糊
    final_img = gaussian_filter(defect_img, sigma=0.2)
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)
    
    # 保存
    Image.fromarray(final_img).save('final_blurred_defects.jpg')
    
    # 創建詳細的視覺化
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])
    
    # 1. 原始圖片
    ax1.imshow(img_np, cmap='gray')
    ax1.set_title('Original Image', fontsize=14)
    ax1.axis('off')
    
    # 2. 正確的結構標註
    ax2.imshow(img_np, cmap='gray')
    ax2.set_title('Accurate Edge Detection', fontsize=14)
    
    # 標記垂直邊緣（不同顏色表示不同類型）
    for edge in edge_info:
        if edge['type'] == 'dark_to_bright':
            ax2.axvline(x=edge['pos'], color='blue', linewidth=2, alpha=0.8)
        else:
            ax2.axvline(x=edge['pos'], color='red', linewidth=2, alpha=0.8)
    
    # 標記水平亮條
    for band_start, band_end in horizontal_bands:
        rect = patches.Rectangle((0, band_start), img_np.shape[1], band_end-band_start,
                               linewidth=0, facecolor='yellow', alpha=0.3)
        ax2.add_patch(rect)
    
    # 標記邊界
    margin_rect = patches.Rectangle((edge_margin, edge_margin), 
                                   img_np.shape[1]-2*edge_margin, 
                                   img_np.shape[0]-2*edge_margin,
                                   linewidth=2, edgecolor='green', 
                                   facecolor='none', linestyle='--')
    ax2.add_patch(margin_rect)
    
    # 添加圖例
    ax2.text(10, 980, 'Blue: Dark→Bright edge, Red: Bright→Dark edge', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    ax2.axis('off')
    
    # 3. 模糊後
    ax3.imshow(blurred_base, cmap='gray', vmin=0, vmax=255)
    ax3.set_title('After Heavy Blur (sigma=2.5)', fontsize=14)
    ax3.axis('off')
    
    # 4. 最終結果（大圖）
    ax4.imshow(final_img, cmap='gray')
    ax4.set_title(f'Final Result: {len(defect_info)} Defects at Correct Edge Positions', fontsize=16)
    
    # 標註缺陷
    for defect in defect_info:
        cx, cy = defect['center']
        w, h = defect['bbox'][2], defect['bbox'][3]
        
        # 橢圓
        ellipse = patches.Ellipse((cx, cy), w, h, linewidth=2.5,
                                edgecolor='red', facecolor='none')
        ax4.add_patch(ellipse)
        
        # 標籤
        edge_type_short = 'D→B' if defect['edge_type'] == 'dark_to_bright' else 'B→D'
        label = f"D{defect['id']}\n×{defect['intensity_factor']:.1f}\n{edge_type_short}"
        ax4.text(cx, cy - h/2 - 25, label, 
                color='red', fontsize=10, ha='center', weight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.9))
    
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('final_blurred_defects_annotated.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 打印總結
    print("\n" + "="*80)
    print("FINAL DEFECT CREATION SUMMARY")
    print("="*80)
    print(f"Successfully created {len(defect_info)} defects")
    print("\nDefects are correctly placed at:")
    print("- Vertical edge positions (where white strips meet dark regions)")
    print("- Horizontal bright band positions")
    print("- Within 100px margins from image borders")
    
    print("\nDefect details:")
    for defect in defect_info:
        cx, cy = defect['center']
        edge_type = "Dark→Bright" if defect['edge_type'] == 'dark_to_bright' else "Bright→Dark"
        print(f"  D{defect['id']:2d}: ({cx:3d}, {cy:3d}) - {edge_type} edge, "
              f"Intensity ×{defect['intensity_factor']:.1f}, Contrast {defect['contrast']:.1f}")
    
    # 保存詳細信息
    with open('final_blurred_defects_info.txt', 'w') as f:
        f.write("Final Blurred Edge Defects with Accurate Positioning\n")
        f.write("="*80 + "\n\n")
        f.write("This is the final test image with:\n")
        f.write("- Correctly detected vertical edges\n")
        f.write("- Heavy Gaussian blur (sigma=2.5)\n")
        f.write("- Elliptical defects at edge+band intersections\n")
        f.write(f"- {edge_margin}px margin from borders\n\n")
        
        for defect in defect_info:
            f.write(f"\nDefect {defect['id']}:\n")
            f.write(f"  Position: {defect['center']}\n")
            f.write(f"  Bbox: {defect['bbox']}\n")
            f.write(f"  Edge type: {defect['edge_type']}\n")
            f.write(f"  Intensity: ×{defect['intensity_factor']:.1f}\n")
            f.write(f"  Local contrast: {defect['contrast']:.1f}\n")
            f.write(f"  Band: y={defect['band'][0]}-{defect['band'][1]}\n")
    
    print("\nFiles created:")
    print("- final_blurred_defects.jpg")
    print("- final_blurred_defects_annotated.png")
    print("- final_blurred_defects_info.txt")
    
    return defect_info


if __name__ == "__main__":
    create_final_realistic_defects()