#!/usr/bin/env python3
"""
創建真實的 SEM 缺陷圖片 v2
- 缺陷為橢圓形
- 位於垂直亮條邊緣與較窄橫向亮條的交界處
- 亮橫條分布在整張圖上
"""

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import os


def analyze_image_structure_v2(image_np):
    """改進的圖片結構分析，找出垂直亮條邊緣和較窄的橫向亮條"""
    print("Analyzing image structure (improved)...")
    
    # 1. 找出垂直亮條及其邊緣
    column_means = np.mean(image_np, axis=0)
    col_gradient = np.abs(np.gradient(column_means))
    
    # 找出梯度峰值（邊緣）
    edge_threshold = np.percentile(col_gradient, 90)
    vertical_edges = []
    
    # 檢測邊緣
    for i in range(1, len(col_gradient)-1):
        if col_gradient[i] > edge_threshold:
            # 確認是從暗到亮或從亮到暗的轉變
            left_mean = np.mean(column_means[max(0, i-10):i])
            right_mean = np.mean(column_means[i:min(len(column_means), i+10)])
            if abs(left_mean - right_mean) > 30:
                vertical_edges.append(i)
    
    # 去除太近的邊緣
    filtered_edges = []
    for edge in vertical_edges:
        if not filtered_edges or min(abs(edge - e) for e in filtered_edges) > 20:
            filtered_edges.append(edge)
    
    print(f"Found {len(filtered_edges)} vertical edges")
    
    # 2. 找出較窄的橫向亮條（整張圖分布）
    bright_bands = []
    
    # 分段分析，確保找到整張圖的亮條
    segment_height = image_np.shape[0] // 10
    
    for seg in range(10):
        start_row = seg * segment_height
        end_row = min((seg + 1) * segment_height, image_np.shape[0])
        
        # 在每個段中找亮條
        segment = image_np[start_row:end_row, :]
        row_means = np.mean(segment, axis=1)
        
        # 使用局部閾值
        local_threshold = np.percentile(row_means, 80)
        
        # 找出連續的亮行
        in_band = False
        band_start = 0
        
        for i, mean_val in enumerate(row_means):
            if mean_val > local_threshold and not in_band:
                in_band = True
                band_start = i
            elif mean_val <= local_threshold and in_band:
                in_band = False
                band_width = i - band_start
                if 5 <= band_width <= 30:  # 亮條寬度在5-30像素之間
                    bright_bands.append((start_row + band_start, start_row + i))
        
        # 處理段末尾的情況
        if in_band:
            band_width = len(row_means) - band_start
            if 5 <= band_width <= 30:
                bright_bands.append((start_row + band_start, end_row))
    
    print(f"Found {len(bright_bands)} narrow bright bands across the image")
    
    return filtered_edges, bright_bands


def create_elliptical_defect(image_region, intensity_factor):
    """創建橢圓形缺陷"""
    h, w = image_region.shape
    
    # 創建橢圓遮罩
    y, x = np.ogrid[:h, :w]
    cy, cx = h/2, w/2
    
    # 橢圓參數（垂直方向稍長）
    a = w/2 * 0.8  # 水平半徑
    b = h/2 * 0.9  # 垂直半徑
    
    # 橢圓方程
    mask = ((x - cx)/a)**2 + ((y - cy)/b)**2 <= 1
    
    # 創建漸變效果（中心最亮）
    distance = np.sqrt(((x - cx)/a)**2 + ((y - cy)/b)**2)
    gradient = np.clip(1 - distance, 0, 1)
    
    # 應用缺陷
    base_brightness = np.mean(image_region)
    defect_brightness = base_brightness * intensity_factor
    
    # 添加一些隨機變化使其更自然
    noise = np.random.normal(0, 3, image_region.shape)
    
    # 組合缺陷
    defect_region = image_region.copy()
    defect_region[mask] = defect_region[mask] * (1 - gradient[mask]) + \
                          (defect_brightness + noise[mask]) * gradient[mask]
    
    return np.clip(defect_region, 0, 255)


def find_valid_defect_locations_v2(image_np, vertical_edges, bright_bands, edge_margin=100):
    """找出有效的缺陷位置（垂直邊緣與橫向亮條的交點）"""
    valid_locations = []
    
    for v_edge in vertical_edges:
        # 跳過太靠近圖片邊緣的垂直邊（使用edge_margin）
        if v_edge < edge_margin or v_edge > image_np.shape[1] - edge_margin:
            continue
            
        for band_start, band_end in bright_bands:
            # 在亮條中心位置
            band_center = (band_start + band_end) // 2
            
            # 檢查這個位置的有效性（確保不在邊緣區域）
            if edge_margin < band_center < image_np.shape[0] - edge_margin:
                # 計算局部對比度
                left_region = image_np[band_center-5:band_center+5, v_edge-15:v_edge-5]
                right_region = image_np[band_center-5:band_center+5, v_edge+5:v_edge+15]
                
                if left_region.size > 0 and right_region.size > 0:
                    contrast = abs(np.mean(left_region) - np.mean(right_region))
                    
                    if contrast > 20:  # 足夠的對比度
                        valid_locations.append({
                            'x': v_edge,
                            'y': band_center,
                            'band_width': band_end - band_start,
                            'contrast': contrast
                        })
    
    # 過濾太近的位置
    filtered_locations = []
    for loc in valid_locations:
        too_close = False
        for existing in filtered_locations:
            if abs(loc['x'] - existing['x']) < 80 and abs(loc['y'] - existing['y']) < 80:
                too_close = True
                break
        if not too_close:
            filtered_locations.append(loc)
    
    # 按y座標排序，確保缺陷分布在整張圖上
    filtered_locations.sort(key=lambda x: x['y'])
    
    print(f"Found {len(filtered_locations)} valid defect locations (excluding {edge_margin}px margins)")
    return filtered_locations


def create_realistic_defects_v2():
    """創建真實的橢圓形缺陷"""
    # 讀取原始圖片
    input_path = 'Img_preprocess/sem_noisy_output_raw.jpg'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        return
    
    img = Image.open(input_path).convert('L')
    img_np = np.array(img)
    
    # 分析圖片結構
    vertical_edges, bright_bands = analyze_image_structure_v2(img_np)
    
    # 找出有效的缺陷位置（排除邊緣100像素）
    edge_margin = 100
    valid_locations = find_valid_defect_locations_v2(img_np, vertical_edges, bright_bands, edge_margin)
    
    if len(valid_locations) == 0:
        print(f"No valid defect locations found with edge_margin={edge_margin}!")
        print("Try reducing edge_margin or check image structure.")
        return
    
    # 選擇缺陷位置（確保分布均勻）
    num_defects = min(10, len(valid_locations))
    if num_defects == 0:
        print("No valid defect locations found!")
        return
    
    # 均勻選擇位置
    step = len(valid_locations) // num_defects
    selected_indices = [i * step for i in range(num_defects)]
    selected_locations = [valid_locations[i] for i in selected_indices if i < len(valid_locations)]
    
    # 創建缺陷
    defect_img = img_np.copy().astype(float)
    defect_info = []
    
    # 不同的缺陷強度
    intensity_factors = np.linspace(1.2, 2.0, len(selected_locations))
    
    for i, (loc, intensity) in enumerate(zip(selected_locations, intensity_factors)):
        x, y = loc['x'], loc['y']
        
        # 橢圓形缺陷大小（垂直方向稍長）
        w = 8 + np.random.randint(-2, 3)
        h = int(w * (1.5 + np.random.uniform(-0.3, 0.3)))  # 高度是寬度的1.2-1.8倍
        
        # 確保不超出邊界
        x1 = max(0, x - w//2)
        y1 = max(0, y - h//2)
        x2 = min(img_np.shape[1], x1 + w)
        y2 = min(img_np.shape[0], y1 + h)
        
        # 獲取區域並創建橢圓形缺陷
        region = defect_img[y1:y2, x1:x2]
        defect_region = create_elliptical_defect(region, intensity)
        defect_img[y1:y2, x1:x2] = defect_region
        
        defect_info.append({
            'id': i + 1,
            'bbox': (x1, y1, x2-x1, y2-y1),
            'center': (x, y),
            'intensity_factor': intensity,
            'contrast': loc['contrast']
        })
    
    # 應用輕微的高斯模糊
    defect_img = gaussian_filter(defect_img, sigma=0.6)
    final_img = np.clip(defect_img, 0, 255).astype(np.uint8)
    
    # 保存結果
    Image.fromarray(final_img).save('realistic_defects_v2.jpg')
    
    # 創建視覺化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左圖：顯示結構分析
    ax1.imshow(img_np, cmap='gray')
    ax1.set_title('Structure Analysis: Vertical Edges + Narrow Bright Bands (100px margins excluded)', fontsize=14)
    
    # 畫出邊緣排除區域
    margin_color = 'blue'
    margin_alpha = 0.1
    # 左邊界
    ax1.axvspan(0, edge_margin, alpha=margin_alpha, color=margin_color)
    # 右邊界
    ax1.axvspan(img_np.shape[1]-edge_margin, img_np.shape[1], alpha=margin_alpha, color=margin_color)
    # 上邊界
    ax1.axhspan(0, edge_margin, alpha=margin_alpha, color=margin_color)
    # 下邊界
    ax1.axhspan(img_np.shape[0]-edge_margin, img_np.shape[0], alpha=margin_alpha, color=margin_color)
    
    # 標記邊界線
    ax1.axvline(x=edge_margin, color='blue', linewidth=1, alpha=0.5, linestyle=':')
    ax1.axvline(x=img_np.shape[1]-edge_margin, color='blue', linewidth=1, alpha=0.5, linestyle=':')
    ax1.axhline(y=edge_margin, color='blue', linewidth=1, alpha=0.5, linestyle=':')
    ax1.axhline(y=img_np.shape[0]-edge_margin, color='blue', linewidth=1, alpha=0.5, linestyle=':')
    
    # 標記垂直邊緣
    for v_edge in vertical_edges:
        ax1.axvline(x=v_edge, color='red', linewidth=1, alpha=0.7, linestyle='--')
    
    # 標記橫向亮條（較窄）
    for band_start, band_end in bright_bands:
        rect = patches.Rectangle((0, band_start), img_np.shape[1], band_end-band_start,
                               linewidth=0, facecolor='yellow', alpha=0.3)
        ax1.add_patch(rect)
    
    # 標記選中的缺陷位置
    for loc in selected_locations:
        circle = plt.Circle((loc['x'], loc['y']), 20, color='lime', 
                          fill=False, linewidth=2)
        ax1.add_patch(circle)
    
    ax1.axis('off')
    
    # 右圖：最終缺陷圖
    ax2.imshow(final_img, cmap='gray')
    ax2.set_title(f'Realistic Elliptical Defects ({len(defect_info)} defects)', fontsize=14)
    
    # 標註缺陷（橢圓形）
    for defect in defect_info:
        x, y, w, h = defect['bbox']
        cx, cy = defect['center']
        
        # 畫橢圓
        ellipse = patches.Ellipse((cx, cy), w, h, linewidth=2,
                                edgecolor='red', facecolor='none')
        ax2.add_patch(ellipse)
        
        # 添加標籤
        label = f"D{defect['id']}\n×{defect['intensity_factor']:.1f}"
        ax2.text(cx, cy - h/2 - 10, label, 
                color='red', fontsize=9, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('realistic_defects_v2_annotated.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 打印缺陷信息
    print("\nDefect Information:")
    print("=" * 70)
    print(f"{'ID':<4} {'Position':<15} {'Size':<12} {'Intensity':<10} {'Y-coord':<10}")
    print("-" * 70)
    
    # 保存缺陷信息到文件
    with open('realistic_defect_locations_v2.txt', 'w') as f:
        f.write("Realistic Elliptical Defect Locations v2\n")
        f.write("=" * 70 + "\n")
        f.write("Defects at: Vertical edge + Narrow bright horizontal band\n")
        f.write(f"Edge margin: {edge_margin}px excluded from all sides\n\n")
        
        for defect in defect_info:
            x, y, w, h = defect['bbox']
            cx, cy = defect['center']
            print(f"D{defect['id']:<3} ({cx:>3}, {cy:>3})      {w:>2} × {h:>2}      "
                  f"×{defect['intensity_factor']:.1f}        y={cy:>3}")
            
            f.write(f"Defect {defect['id']}:\n")
            f.write(f"  Bbox (x, y, w, h): ({x}, {y}, {w}, {h})\n")
            f.write(f"  Center: ({cx}, {cy})\n")
            f.write(f"  Intensity factor: ×{defect['intensity_factor']:.1f}\n")
            f.write(f"  Edge contrast: {defect['contrast']:.1f}\n\n")
    
    print(f"\nDefects distributed from y={min(d['center'][1] for d in defect_info)} "
          f"to y={max(d['center'][1] for d in defect_info)}")
    print("\nFiles created:")
    print("- realistic_defects_v2.jpg")
    print("- realistic_defects_v2_annotated.png")
    print("- realistic_defect_locations_v2.txt")


if __name__ == "__main__":
    create_realistic_defects_v2()