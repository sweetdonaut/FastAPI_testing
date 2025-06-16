#!/usr/bin/env python3
"""
創建邊緣模糊的 SEM 缺陷圖片
先對原始圖片進行強模糊處理，使邊緣變得不清晰
然後在模糊的邊緣上添加缺陷
"""

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import os


def create_blurred_base_image(image_np, blur_sigma=2.5):
    """對原始圖片進行模糊處理，使邊緣變得不清晰"""
    print(f"Applying edge blur with sigma={blur_sigma}...")
    
    # 應用較強的高斯模糊
    blurred = gaussian_filter(image_np.astype(float), sigma=blur_sigma)
    
    # 增加一些雜訊使圖片更真實
    noise = np.random.normal(0, 1, blurred.shape)
    blurred_noisy = blurred + noise
    
    return np.clip(blurred_noisy, 0, 255)


def analyze_blurred_structure(blurred_img):
    """分析模糊圖片的結構（邊緣會更難檢測）"""
    print("Analyzing blurred image structure...")
    
    # 1. 嘗試找出模糊的垂直邊緣
    column_means = np.mean(blurred_img, axis=0)
    col_gradient = np.abs(np.gradient(column_means))
    
    # 由於模糊，需要降低閾值
    edge_threshold = np.percentile(col_gradient, 85)  # 降低閾值
    vertical_edges = []
    
    for i in range(10, len(col_gradient)-10):
        if col_gradient[i] > edge_threshold:
            # 使用更寬的窗口來計算對比度
            left_mean = np.mean(column_means[max(0, i-20):i-5])
            right_mean = np.mean(column_means[i+5:min(len(column_means), i+20)])
            if abs(left_mean - right_mean) > 20:  # 降低對比度要求
                vertical_edges.append(i)
    
    # 去除太近的邊緣
    filtered_edges = []
    for edge in vertical_edges:
        if not filtered_edges or min(abs(edge - e) for e in filtered_edges) > 30:
            filtered_edges.append(edge)
    
    print(f"Found {len(filtered_edges)} blurred vertical edges")
    
    # 2. 找出模糊的橫向亮條
    bright_bands = []
    segment_height = blurred_img.shape[0] // 8
    
    for seg in range(8):
        start_row = seg * segment_height
        end_row = min((seg + 1) * segment_height, blurred_img.shape[0])
        
        segment = blurred_img[start_row:end_row, :]
        row_means = np.mean(segment, axis=1)
        
        # 使用更寬鬆的閾值
        local_threshold = np.percentile(row_means, 75)
        
        # 找出亮條（可能更寬因為模糊）
        in_band = False
        band_start = 0
        
        for i, mean_val in enumerate(row_means):
            if mean_val > local_threshold and not in_band:
                in_band = True
                band_start = i
            elif mean_val <= local_threshold and in_band:
                in_band = False
                band_width = i - band_start
                if 10 <= band_width <= 40:  # 允許更寬的條帶
                    bright_bands.append((start_row + band_start, start_row + i))
        
        if in_band:
            band_width = len(row_means) - band_start
            if 10 <= band_width <= 40:
                bright_bands.append((start_row + band_start, end_row))
    
    print(f"Found {len(bright_bands)} blurred bright bands")
    
    return filtered_edges, bright_bands


def create_elliptical_defect_on_blur(image_region, intensity_factor):
    """在模糊背景上創建橢圓形缺陷"""
    h, w = image_region.shape
    
    # 創建橢圓遮罩
    y, x = np.ogrid[:h, :w]
    cy, cx = h/2, w/2
    
    # 橢圓參數
    a = w/2 * 0.7
    b = h/2 * 0.8
    
    # 橢圓方程
    mask = ((x - cx)/a)**2 + ((y - cy)/b)**2 <= 1
    
    # 創建更柔和的漸變（適應模糊背景）
    distance = np.sqrt(((x - cx)/a)**2 + ((y - cy)/b)**2)
    gradient = np.clip(1 - distance * 0.8, 0, 1)  # 更柔和的漸變
    
    # 應用缺陷
    base_brightness = np.mean(image_region)
    defect_brightness = base_brightness * intensity_factor
    
    # 較少的雜訊，因為背景已經模糊
    noise = np.random.normal(0, 2, image_region.shape)
    
    # 組合缺陷
    defect_region = image_region.copy()
    defect_region[mask] = defect_region[mask] * (1 - gradient[mask] * 0.7) + \
                          (defect_brightness + noise[mask]) * gradient[mask] * 0.7
    
    return np.clip(defect_region, 0, 255)


def create_blurred_edge_defects():
    """創建邊緣模糊的缺陷圖片"""
    # 讀取原始圖片
    input_path = 'Img_preprocess/sem_noisy_output_raw.jpg'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        return
    
    img = Image.open(input_path).convert('L')
    img_np = np.array(img)
    
    # 步驟1：創建模糊的基礎圖片
    blurred_base = create_blurred_base_image(img_np, blur_sigma=2.5)
    
    # 步驟2：分析模糊圖片的結構
    vertical_edges, bright_bands = analyze_blurred_structure(blurred_base)
    
    # 步驟3：找出有效的缺陷位置
    edge_margin = 100
    valid_locations = []
    
    for v_edge in vertical_edges:
        if v_edge < edge_margin or v_edge > blurred_base.shape[1] - edge_margin:
            continue
            
        for band_start, band_end in bright_bands:
            band_center = (band_start + band_end) // 2
            
            if edge_margin < band_center < blurred_base.shape[0] - edge_margin:
                # 在模糊圖片上計算對比度
                left_region = blurred_base[band_center-10:band_center+10, v_edge-20:v_edge-5]
                right_region = blurred_base[band_center-10:band_center+10, v_edge+5:v_edge+20]
                
                if left_region.size > 0 and right_region.size > 0:
                    contrast = abs(np.mean(left_region) - np.mean(right_region))
                    
                    if contrast > 15:  # 降低對比度要求
                        valid_locations.append({
                            'x': v_edge,
                            'y': band_center,
                            'contrast': contrast
                        })
    
    # 過濾太近的位置
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
    
    print(f"Found {len(filtered_locations)} valid locations on blurred edges")
    
    if len(filtered_locations) == 0:
        print("No valid defect locations found on blurred image!")
        return
    
    # 步驟4：添加缺陷到模糊圖片
    defect_img = blurred_base.copy()
    defect_info = []
    
    # 選擇缺陷位置
    num_defects = min(8, len(filtered_locations))
    step = max(1, len(filtered_locations) // num_defects)
    selected_locations = filtered_locations[::step][:num_defects]
    
    # 缺陷強度
    intensity_factors = np.linspace(1.3, 2.2, len(selected_locations))  # 需要更高的強度
    
    for i, (loc, intensity) in enumerate(zip(selected_locations, intensity_factors)):
        x, y = loc['x'], loc['y']
        
        # 橢圓形缺陷
        w = 10 + np.random.randint(-2, 3)
        h = int(w * (1.4 + np.random.uniform(-0.2, 0.2)))
        
        x1 = max(0, x - w//2)
        y1 = max(0, y - h//2)
        x2 = min(defect_img.shape[1], x1 + w)
        y2 = min(defect_img.shape[0], y1 + h)
        
        # 創建缺陷
        region = defect_img[y1:y2, x1:x2]
        if region.size > 0:
            defect_region = create_elliptical_defect_on_blur(region, intensity)
            defect_img[y1:y2, x1:x2] = defect_region
            
            defect_info.append({
                'id': i + 1,
                'bbox': (x1, y1, x2-x1, y2-y1),
                'center': (x, y),
                'intensity_factor': intensity,
                'contrast': loc['contrast']
            })
    
    # 步驟5：最後再應用輕微模糊
    final_img = gaussian_filter(defect_img, sigma=0.3)
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)
    
    # 保存結果
    Image.fromarray(final_img).save('blurred_edge_defects.jpg')
    
    # 創建視覺化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
    
    # 原始圖片
    ax1.imshow(img_np, cmap='gray')
    ax1.set_title('Original Image', fontsize=14)
    ax1.axis('off')
    
    # 模糊後的基礎圖片
    ax2.imshow(blurred_base, cmap='gray', vmin=0, vmax=255)
    ax2.set_title(f'Blurred Base (sigma=2.5)', fontsize=14)
    ax2.axis('off')
    
    # 結構分析
    ax3.imshow(blurred_base, cmap='gray', vmin=0, vmax=255)
    ax3.set_title('Detected Structure on Blurred Image', fontsize=14)
    
    # 標記檢測到的邊緣和亮條
    for v_edge in vertical_edges:
        ax3.axvline(x=v_edge, color='red', linewidth=1, alpha=0.7, linestyle='--')
    
    for band_start, band_end in bright_bands:
        rect = patches.Rectangle((0, band_start), blurred_base.shape[1], band_end-band_start,
                               linewidth=0, facecolor='yellow', alpha=0.2)
        ax3.add_patch(rect)
    
    # 標記缺陷位置
    for defect in defect_info:
        cx, cy = defect['center']
        circle = plt.Circle((cx, cy), 15, color='lime', fill=False, linewidth=2)
        ax3.add_patch(circle)
    
    ax3.axis('off')
    
    # 最終缺陷圖
    ax4.imshow(final_img, cmap='gray')
    ax4.set_title(f'Final Image with {len(defect_info)} Defects on Blurred Edges', fontsize=14)
    
    # 標註缺陷
    for defect in defect_info:
        x, y, w, h = defect['bbox']
        cx, cy = defect['center']
        
        ellipse = patches.Ellipse((cx, cy), w, h, linewidth=2,
                                edgecolor='red', facecolor='none')
        ax4.add_patch(ellipse)
        
        label = f"D{defect['id']}\n×{defect['intensity_factor']:.1f}"
        ax4.text(cx, cy - h/2 - 15, label, 
                color='red', fontsize=9, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('blurred_edge_defects_annotated.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存缺陷信息
    with open('blurred_edge_defect_locations.txt', 'w') as f:
        f.write("Defect Locations on Blurred Edges\n")
        f.write("=" * 70 + "\n")
        f.write("Challenge: Edges are heavily blurred (sigma=2.5)\n")
        f.write("Higher intensity factors needed due to reduced contrast\n\n")
        
        for defect in defect_info:
            x, y, w, h = defect['bbox']
            cx, cy = defect['center']
            f.write(f"Defect {defect['id']}:\n")
            f.write(f"  Bbox: ({x}, {y}, {w}, {h})\n")
            f.write(f"  Center: ({cx}, {cy})\n")
            f.write(f"  Intensity: ×{defect['intensity_factor']:.1f}\n")
            f.write(f"  Local contrast: {defect['contrast']:.1f}\n\n")
    
    print("\nFiles created:")
    print("- blurred_edge_defects.jpg")
    print("- blurred_edge_defects_annotated.png")
    print("- blurred_edge_defect_locations.txt")
    
    print(f"\nCreated {len(defect_info)} defects on blurred edges")
    print("Note: Detection will be more challenging due to:")
    print("- Heavily blurred edges (sigma=2.5)")
    print("- Reduced contrast between regions")
    print("- Less defined grid structure")


if __name__ == "__main__":
    create_blurred_edge_defects()