#!/usr/bin/env python3
"""
創建邊緣模糊的 SEM 缺陷圖片 v2
修正垂直邊緣檢測的歪斜問題
使用更穩健的邊緣檢測方法
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter, sobel
from scipy import signal
import os


def detect_image_rotation(image_np):
    """檢測圖片是否有輕微旋轉"""
    # 使用 Sobel 檢測垂直邊緣
    edges_vertical = sobel(image_np, axis=1)
    
    # 計算邊緣的角度分布
    # 這裡簡化處理，假設圖片基本是正的
    return 0  # 假設沒有旋轉


def detect_vertical_edges_robust(image_np):
    """使用更穩健的方法檢測垂直邊緣"""
    print("Detecting vertical edges with improved method...")
    
    # 方法1：使用垂直 Sobel 濾波器
    vertical_edges_sobel = sobel(image_np, axis=1)
    
    # 對每一列計算邊緣強度
    edge_strength = np.mean(np.abs(vertical_edges_sobel), axis=0)
    
    # 方法2：使用局部窗口的梯度
    # 為了處理可能的輕微傾斜，使用稍寬的窗口
    window_width = 5
    local_gradients = []
    
    for i in range(window_width, image_np.shape[1] - window_width):
        # 計算局部區域的水平梯度
        left_region = image_np[:, i-window_width:i]
        right_region = image_np[:, i:i+window_width]
        
        left_mean = np.mean(left_region, axis=1)
        right_mean = np.mean(right_region, axis=1)
        
        # 計算整體梯度（考慮所有行）
        gradient = np.mean(np.abs(right_mean - left_mean))
        local_gradients.append(gradient)
    
    local_gradients = np.array(local_gradients)
    
    # 結合兩種方法
    # 找出梯度峰值
    threshold = np.percentile(local_gradients, 85)
    
    vertical_edges = []
    for i in range(len(local_gradients)):
        if local_gradients[i] > threshold:
            actual_pos = i + window_width
            
            # 驗證這是一個真實的邊緣
            if actual_pos > 20 and actual_pos < image_np.shape[1] - 20:
                # 計算更大範圍的對比度
                left_val = np.mean(image_np[:, actual_pos-20:actual_pos-5])
                right_val = np.mean(image_np[:, actual_pos+5:actual_pos+20])
                
                if abs(left_val - right_val) > 25:
                    vertical_edges.append(actual_pos)
    
    # 去除太近的邊緣
    filtered_edges = []
    for edge in vertical_edges:
        if not filtered_edges or min(abs(edge - e) for e in filtered_edges) > 40:
            filtered_edges.append(edge)
    
    # 確保邊緣是垂直的（通過檢查多個高度位置）
    verified_edges = []
    for edge in filtered_edges:
        # 在不同高度檢查邊緣位置
        positions = []
        for y in range(100, image_np.shape[0] - 100, 100):
            # 在這個高度找局部最大梯度
            local_region = image_np[y-50:y+50, max(0, edge-30):min(image_np.shape[1], edge+30)]
            if local_region.shape[1] > 0:
                local_grad = np.diff(np.mean(local_region, axis=0))
                if len(local_grad) > 0:
                    max_pos = np.argmax(np.abs(local_grad))
                    positions.append(edge - 30 + max_pos)
        
        # 如果位置變化不大，說明是垂直的
        if len(positions) > 0 and np.std(positions) < 5:
            verified_edges.append(int(np.mean(positions)))
    
    return verified_edges


def detect_horizontal_bands_robust(image_np):
    """檢測水平亮條，考慮可能的不均勻性"""
    print("Detecting horizontal bands...")
    
    bright_bands = []
    
    # 使用滑動窗口方法
    window_height = 20  # 縮小窗口
    step = 5  # 更細的步長
    
    for y in range(0, image_np.shape[0] - window_height, step):
        window = image_np[y:y+window_height, :]
        mean_brightness = np.mean(window)
        
        # 檢查是否是亮條
        if mean_brightness > np.percentile(image_np, 65):  # 降低閾值
            # 檢查是否與已有條帶重疊
            overlapping = False
            for start, end in bright_bands:
                if y < end and y + window_height > start:
                    overlapping = True
                    break
            
            if not overlapping:
                bright_bands.append((y, y + window_height))
    
    # 合併相鄰的亮條
    merged_bands = []
    for start, end in sorted(bright_bands):
        if merged_bands and start <= merged_bands[-1][1] + 10:
            merged_bands[-1] = (merged_bands[-1][0], max(end, merged_bands[-1][1]))
        else:
            merged_bands.append((start, end))
    
    # 過濾太窄或太寬的條帶
    final_bands = []
    for start, end in merged_bands:
        width = end - start
        if 10 <= width <= 60:  # 放寬條件
            final_bands.append((start, end))
    
    return final_bands


def create_blurred_base_image_v2(image_np, blur_sigma=2.5):
    """創建模糊的基礎圖片"""
    print(f"Creating blurred base image (sigma={blur_sigma})...")
    
    # 應用高斯模糊
    blurred = gaussian_filter(image_np.astype(float), sigma=blur_sigma)
    
    # 添加輕微的雜訊
    noise = np.random.normal(0, 1, blurred.shape)
    blurred_noisy = blurred + noise
    
    return np.clip(blurred_noisy, 0, 255)


def create_elliptical_defect_soft(image_region, intensity_factor):
    """創建柔和的橢圓形缺陷"""
    h, w = image_region.shape
    
    if h < 3 or w < 3:
        return image_region
    
    # 創建橢圓遮罩
    y, x = np.ogrid[:h, :w]
    cy, cx = h/2, w/2
    
    # 橢圓參數
    a = max(1, w/2 * 0.7)
    b = max(1, h/2 * 0.8)
    
    # 橢圓方程與柔和邊緣
    distance = np.sqrt(((x - cx)/a)**2 + ((y - cy)/b)**2)
    mask = distance <= 1.2  # 稍大的遮罩範圍
    gradient = np.exp(-2 * distance)  # 指數衰減，更自然
    
    # 創建缺陷
    base_brightness = np.mean(image_region)
    defect_brightness = base_brightness * intensity_factor
    
    # 少量雜訊
    noise = np.random.normal(0, 2, image_region.shape)
    
    # 應用缺陷
    defect_region = image_region.copy()
    blend_factor = gradient * 0.6  # 降低混合強度
    defect_region = defect_region * (1 - blend_factor) + (defect_brightness + noise) * blend_factor
    
    return np.clip(defect_region, 0, 255)


def create_blurred_edge_defects_v2():
    """創建邊緣模糊的缺陷圖片，修正垂直邊緣檢測"""
    # 讀取原始圖片
    input_path = 'Img_preprocess/sem_noisy_output_raw.jpg'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        return
    
    img = Image.open(input_path).convert('L')
    img_np = np.array(img)
    
    # 步驟1：創建模糊的基礎圖片
    blurred_base = create_blurred_base_image_v2(img_np, blur_sigma=2.5)
    
    # 步驟2：使用改進的方法檢測結構
    # 注意：在原始圖片上檢測，然後應用到模糊圖片
    print("Detecting structure on original image first...")
    vertical_edges_orig = detect_vertical_edges_robust(img_np)
    horizontal_bands_orig = detect_horizontal_bands_robust(img_np)
    
    print(f"Found {len(vertical_edges_orig)} vertical edges on original")
    print(f"Found {len(horizontal_bands_orig)} horizontal bands on original")
    
    # 步驟3：找出有效的缺陷位置
    edge_margin = 100
    valid_locations = []
    
    for v_edge in vertical_edges_orig:
        if v_edge < edge_margin or v_edge > img_np.shape[1] - edge_margin:
            continue
        
        for band_start, band_end in horizontal_bands_orig:
            band_center = (band_start + band_end) // 2
            
            if edge_margin < band_center < img_np.shape[0] - edge_margin:
                # 在模糊圖片上驗證對比度
                left_region = blurred_base[band_center-10:band_center+10, v_edge-20:v_edge-5]
                right_region = blurred_base[band_center-10:band_center+10, v_edge+5:v_edge+20]
                
                if left_region.size > 0 and right_region.size > 0:
                    contrast = abs(np.mean(left_region) - np.mean(right_region))
                    
                    if contrast > 10:  # 在模糊圖上仍有足夠對比度
                        valid_locations.append({
                            'x': v_edge,
                            'y': band_center,
                            'contrast': contrast,
                            'band_width': band_end - band_start
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
    
    # 按y坐標排序
    filtered_locations.sort(key=lambda x: x['y'])
    
    print(f"Found {len(filtered_locations)} valid defect locations")
    
    if len(filtered_locations) == 0:
        print("No valid defect locations found!")
        return
    
    # 步驟4：添加缺陷
    defect_img = blurred_base.copy()
    defect_info = []
    
    # 選擇缺陷位置
    num_defects = min(10, len(filtered_locations))
    step = max(1, len(filtered_locations) // num_defects)
    selected_locations = filtered_locations[::step][:num_defects]
    
    # 缺陷強度
    intensity_factors = np.linspace(1.3, 2.2, len(selected_locations))
    
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
            defect_region = create_elliptical_defect_soft(region, intensity)
            defect_img[y1:y2, x1:x2] = defect_region
            
            defect_info.append({
                'id': i + 1,
                'bbox': (x1, y1, x2-x1, y2-y1),
                'center': (x, y),
                'intensity_factor': intensity,
                'contrast': loc['contrast']
            })
    
    # 最後的輕微模糊
    final_img = gaussian_filter(defect_img, sigma=0.3)
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)
    
    # 保存結果
    Image.fromarray(final_img).save('blurred_edge_defects_v2.jpg')
    
    # 創建視覺化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
    
    # 原始圖片
    ax1.imshow(img_np, cmap='gray')
    ax1.set_title('Original Image', fontsize=14)
    ax1.axis('off')
    
    # 原始圖片上的邊緣檢測
    ax2.imshow(img_np, cmap='gray')
    ax2.set_title('Improved Edge Detection (on Original)', fontsize=14)
    
    # 顯示垂直邊緣（應該是直的）
    for v_edge in vertical_edges_orig:
        ax2.axvline(x=v_edge, color='red', linewidth=1.5, alpha=0.8)
    
    # 顯示水平條帶
    for band_start, band_end in horizontal_bands_orig:
        rect = patches.Rectangle((0, band_start), img_np.shape[1], band_end-band_start,
                               linewidth=0, facecolor='yellow', alpha=0.3)
        ax2.add_patch(rect)
    
    ax2.axis('off')
    
    # 模糊圖片
    ax3.imshow(blurred_base, cmap='gray', vmin=0, vmax=255)
    ax3.set_title(f'Blurred Base (sigma=2.5)', fontsize=14)
    ax3.axis('off')
    
    # 最終缺陷圖
    ax4.imshow(final_img, cmap='gray')
    ax4.set_title(f'Final: {len(defect_info)} Defects on Blurred Edges', fontsize=14)
    
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
    plt.savefig('blurred_edge_defects_v2_annotated.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存缺陷信息
    print("\nDefect Information:")
    print("=" * 70)
    print(f"{'ID':<4} {'Position':<15} {'Size':<12} {'Intensity':<10} {'Contrast':<10}")
    print("-" * 70)
    
    with open('blurred_edge_defect_locations_v2.txt', 'w') as f:
        f.write("Defect Locations on Blurred Edges (Improved Detection)\n")
        f.write("=" * 70 + "\n")
        f.write("- Vertical edges detected with robust method\n")
        f.write("- Edges verified to be truly vertical\n")
        f.write("- Defects placed at correct edge positions\n\n")
        
        for defect in defect_info:
            x, y, w, h = defect['bbox']
            cx, cy = defect['center']
            
            print(f"D{defect['id']:<3} ({cx:>3}, {cy:>3})      "
                  f"{w:>2} × {h:>2}      ×{defect['intensity_factor']:.1f}      "
                  f"{defect['contrast']:.1f}")
            
            f.write(f"Defect {defect['id']}:\n")
            f.write(f"  Bbox: ({x}, {y}, {w}, {h})\n")
            f.write(f"  Center: ({cx}, {cy})\n")
            f.write(f"  Intensity: ×{defect['intensity_factor']:.1f}\n")
            f.write(f"  Local contrast: {defect['contrast']:.1f}\n\n")
    
    print("\nFiles created:")
    print("- blurred_edge_defects_v2.jpg")
    print("- blurred_edge_defects_v2_annotated.png")
    print("- blurred_edge_defect_locations_v2.txt")


if __name__ == "__main__":
    create_blurred_edge_defects_v2()