#!/usr/bin/env python3
"""
創建真實的 SEM 缺陷圖片
缺陷只出現在：垂直亮條邊緣 + 較亮橫條 的交界處
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import os


def analyze_image_structure(image_np):
    """分析圖片結構，找出垂直亮條和較亮橫條"""
    print("Analyzing image structure...")
    
    # 1. 找出垂直亮條
    column_means = np.mean(image_np, axis=0)
    col_threshold = np.percentile(column_means, 80)  # 前20%最亮的列
    bright_columns = np.where(column_means > col_threshold)[0]
    
    # 找出亮條的邊緣
    vertical_edges = []
    for i in range(1, len(bright_columns)):
        if bright_columns[i] - bright_columns[i-1] > 1:  # 不連續，是邊緣
            vertical_edges.append(bright_columns[i-1])  # 右邊緣
            if i < len(bright_columns) - 1:
                vertical_edges.append(bright_columns[i])  # 左邊緣
    
    print(f"Found {len(vertical_edges)} vertical bright strip edges")
    
    # 2. 找出較亮的橫條
    row_means = np.mean(image_np, axis=1)
    row_threshold = np.percentile(row_means, 70)  # 前30%最亮的行
    bright_rows = []
    
    # 將連續的亮行分組成橫條
    current_band = []
    for i, mean_val in enumerate(row_means):
        if mean_val > row_threshold:
            current_band.append(i)
        else:
            if len(current_band) > 10:  # 至少10個像素寬的橫條
                bright_rows.append((min(current_band), max(current_band)))
            current_band = []
    
    if len(current_band) > 10:
        bright_rows.append((min(current_band), max(current_band)))
    
    print(f"Found {len(bright_rows)} bright horizontal bands")
    
    return vertical_edges, bright_rows


def find_valid_defect_locations(image_np, vertical_edges, bright_rows):
    """找出符合條件的缺陷位置：垂直亮條邊緣 + 較亮橫條"""
    valid_locations = []
    
    for v_edge in vertical_edges:
        for row_start, row_end in bright_rows:
            # 在每個橫條中選擇幾個位置
            row_positions = np.linspace(row_start + 5, row_end - 5, 3, dtype=int)
            
            for row_pos in row_positions:
                # 檢查這個位置是否真的符合條件
                # 垂直邊緣檢查（應該有明顯的亮度變化）
                if v_edge > 10 and v_edge < image_np.shape[1] - 10:
                    left_brightness = np.mean(image_np[row_pos-5:row_pos+5, v_edge-10:v_edge-5])
                    right_brightness = np.mean(image_np[row_pos-5:row_pos+5, v_edge+5:v_edge+10])
                    
                    # 確認是邊緣（亮度差異明顯）
                    if abs(left_brightness - right_brightness) > 20:
                        valid_locations.append({
                            'x': v_edge,
                            'y': row_pos,
                            'edge_contrast': abs(left_brightness - right_brightness),
                            'local_brightness': max(left_brightness, right_brightness)
                        })
    
    # 過濾太近的位置
    filtered_locations = []
    for loc in valid_locations:
        too_close = False
        for existing in filtered_locations:
            if abs(loc['x'] - existing['x']) < 50 and abs(loc['y'] - existing['y']) < 50:
                too_close = True
                break
        if not too_close:
            filtered_locations.append(loc)
    
    print(f"Found {len(filtered_locations)} valid defect locations")
    return filtered_locations


def create_realistic_defects():
    """創建真實的缺陷圖片"""
    # 讀取原始圖片
    input_path = 'Img_preprocess/sem_noisy_output_raw.jpg'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        return
    
    img = Image.open(input_path).convert('L')
    img_np = np.array(img)
    
    # 分析圖片結構
    vertical_edges, bright_rows = analyze_image_structure(img_np)
    
    # 找出有效的缺陷位置
    valid_locations = find_valid_defect_locations(img_np, vertical_edges, bright_rows)
    
    # 選擇一些位置添加缺陷
    num_defects = min(8, len(valid_locations))
    if num_defects == 0:
        print("No valid defect locations found!")
        return
    
    # 排序選擇最佳位置（高對比度邊緣）
    valid_locations.sort(key=lambda x: x['edge_contrast'], reverse=True)
    selected_locations = valid_locations[:num_defects]
    
    # 創建缺陷
    defect_img = img_np.copy()
    defect_info = []
    
    # 不同的缺陷強度
    intensity_factors = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0][:num_defects]
    
    for i, (loc, intensity) in enumerate(zip(selected_locations, intensity_factors)):
        x, y = loc['x'], loc['y']
        
        # 缺陷大小（垂直方向較長，因為是沿著垂直邊緣）
        w = 8 + np.random.randint(-2, 3)
        h = 20 + np.random.randint(-5, 6)
        
        # 確保不超出邊界
        x1 = max(0, x - w//2)
        y1 = max(0, y - h//2)
        x2 = min(img_np.shape[1], x1 + w)
        y2 = min(img_np.shape[0], y1 + h)
        
        # 獲取該區域的平均亮度
        region = defect_img[y1:y2, x1:x2]
        avg_brightness = np.mean(region)
        
        # 創建缺陷（亮點）
        defect_brightness = avg_brightness * intensity
        
        # 添加自然的變化
        noise = np.random.normal(0, 5, region.shape)
        defect_region = np.clip(defect_brightness + noise, 0, 255)
        
        # 應用到圖片
        defect_img[y1:y2, x1:x2] = defect_region
        
        defect_info.append({
            'id': i + 1,
            'bbox': (x1, y1, x2-x1, y2-y1),
            'center': (x, y),
            'intensity_factor': intensity,
            'edge_contrast': loc['edge_contrast'],
            'local_brightness': loc['local_brightness']
        })
    
    # 應用高斯模糊
    blurred_img = gaussian_filter(defect_img, sigma=0.8)
    final_img = np.clip(blurred_img, 0, 255).astype(np.uint8)
    
    # 保存結果
    Image.fromarray(final_img).save('realistic_defects.jpg')
    
    # 創建視覺化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左圖：顯示結構分析
    ax1.imshow(img_np, cmap='gray')
    ax1.set_title('Structure Analysis', fontsize=14)
    
    # 標記垂直邊緣
    for v_edge in vertical_edges:
        ax1.axvline(x=v_edge, color='red', linewidth=1, alpha=0.5)
    
    # 標記亮橫條
    for row_start, row_end in bright_rows:
        rect = patches.Rectangle((0, row_start), img_np.shape[1], row_end-row_start,
                               linewidth=0, facecolor='yellow', alpha=0.2)
        ax1.add_patch(rect)
    
    # 標記有效缺陷位置
    for loc in selected_locations:
        circle = plt.Circle((loc['x'], loc['y']), 15, color='lime', fill=False, linewidth=2)
        ax1.add_patch(circle)
    
    ax1.set_title('Valid Defect Locations (Vertical Edge + Bright Band)', fontsize=12)
    ax1.axis('off')
    
    # 右圖：最終缺陷圖
    ax2.imshow(final_img, cmap='gray')
    ax2.set_title(f'Realistic Defects ({len(defect_info)} defects)', fontsize=14)
    
    # 標註缺陷
    for defect in defect_info:
        x, y, w, h = defect['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor='red', facecolor='none')
        ax2.add_patch(rect)
        
        # 添加標籤
        label = f"D{defect['id']}\n×{defect['intensity_factor']:.1f}"
        ax2.text(x + w/2, y - 10, label, 
                color='red', fontsize=9, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('realistic_defects_annotated.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存缺陷信息
    print("\nDefect Information:")
    print("=" * 70)
    print(f"{'ID':<4} {'Position':<15} {'Size':<10} {'Intensity':<10} {'Edge Contrast':<15}")
    print("-" * 70)
    
    with open('realistic_defect_locations.txt', 'w') as f:
        f.write("Realistic Defect Locations\n")
        f.write("=" * 70 + "\n")
        f.write("Defects at: Vertical bright strip edge + Bright horizontal band\n\n")
        
        for defect in defect_info:
            x, y, w, h = defect['bbox']
            print(f"D{defect['id']:<3} ({x:>3}, {y:>3})      {w:>2} × {h:>2}     ×{defect['intensity_factor']:.1f}        {defect['edge_contrast']:.1f}")
            
            f.write(f"Defect {defect['id']}:\n")
            f.write(f"  Bbox (x, y, w, h): ({x}, {y}, {w}, {h})\n")
            f.write(f"  Center: {defect['center']}\n")
            f.write(f"  Intensity factor: ×{defect['intensity_factor']}\n")
            f.write(f"  Edge contrast: {defect['edge_contrast']:.1f}\n")
            f.write(f"  Local brightness: {defect['local_brightness']:.1f}\n\n")
    
    print("\nFiles created:")
    print("- realistic_defects.jpg")
    print("- realistic_defects_annotated.png")
    print("- realistic_defect_locations.txt")


if __name__ == "__main__":
    create_realistic_defects()