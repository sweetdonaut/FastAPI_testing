#!/usr/bin/env python3
"""
創建更具挑戰性的缺陷圖片
在直條和橫條交界處添加亮點缺陷，並加上高斯模糊
"""

import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import os


def detect_intersections(image_np):
    """檢測直條和橫條的交界處"""
    # 計算梯度找出邊緣
    grad_x = np.gradient(image_np, axis=1)
    grad_y = np.gradient(image_np, axis=0)
    
    # 找出強邊緣
    edge_x = np.abs(grad_x) > np.std(grad_x) * 2
    edge_y = np.abs(grad_y) > np.std(grad_y) * 2
    
    # 找交界點（水平和垂直邊緣都強的地方）
    intersections = edge_x & edge_y
    
    # 找出交界點的座標
    y_coords, x_coords = np.where(intersections)
    
    # 過濾和聚類相近的點
    if len(x_coords) > 0:
        points = []
        for x, y in zip(x_coords, y_coords):
            # 檢查是否與已有點太近
            too_close = False
            for px, py in points:
                if abs(x - px) < 20 and abs(y - py) < 20:
                    too_close = True
                    break
            if not too_close:
                points.append((x, y))
        return points
    return []


def add_defects_at_intersections(image_np, intersection_points, num_defects=5, 
                                defect_size=(8, 15), intensity_factor=1.5):
    """在交界處添加亮點缺陷"""
    # 複製圖片
    defect_image = image_np.copy()
    defect_info = []
    
    # 隨機選擇一些交界點來添加缺陷
    if len(intersection_points) > num_defects:
        import random
        selected_points = random.sample(intersection_points, num_defects)
    else:
        selected_points = intersection_points
    
    for i, (x, y) in enumerate(selected_points):
        # 隨機調整缺陷大小
        w = defect_size[0] + np.random.randint(-2, 3)
        h = defect_size[1] + np.random.randint(-5, 6)
        
        # 確保不超出邊界
        x1 = max(0, x - w//2)
        y1 = max(0, y - h//2)
        x2 = min(image_np.shape[1], x1 + w)
        y2 = min(image_np.shape[0], y1 + h)
        
        # 獲取該區域的平均亮度
        region = defect_image[y1:y2, x1:x2]
        avg_brightness = np.mean(region)
        
        # 創建缺陷（亮點）
        defect_brightness = avg_brightness * intensity_factor
        
        # 添加一些隨機變化，使缺陷更自然
        noise = np.random.normal(0, 5, region.shape)
        defect_region = np.clip(defect_brightness + noise, 0, 255)
        
        # 應用到圖片
        defect_image[y1:y2, x1:x2] = defect_region
        
        # 記錄缺陷位置
        defect_info.append({
            'id': i + 1,
            'bbox': (x1, y1, x2-x1, y2-y1),
            'center': (x, y),
            'brightness': np.mean(defect_region)
        })
    
    return defect_image, defect_info


def create_challenging_defect_image():
    """創建具有挑戰性的缺陷圖片"""
    # 讀取原始圖片
    input_path = 'Img_preprocess/sem_noisy_output_raw.jpg'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        return
    
    # 讀取圖片
    img = Image.open(input_path).convert('L')
    img_np = np.array(img)
    
    print(f"Original image shape: {img_np.shape}")
    
    # 1. 先找出交界點
    print("Detecting intersections...")
    intersection_points = detect_intersections(img_np)
    print(f"Found {len(intersection_points)} intersection points")
    
    # 2. 添加缺陷
    print("Adding defects at intersections...")
    defect_image, defect_info = add_defects_at_intersections(
        img_np, 
        intersection_points, 
        num_defects=8,  # 添加8個缺陷
        defect_size=(10, 20),
        intensity_factor=1.4
    )
    
    # 3. 應用高斯模糊
    print("Applying Gaussian blur...")
    blurred_image = gaussian_filter(defect_image, sigma=0.8)
    
    # 4. 添加一些額外的雜訊使圖片更真實
    noise = np.random.normal(0, 2, blurred_image.shape)
    final_image = np.clip(blurred_image + noise, 0, 255).astype(np.uint8)
    
    # 5. 保存結果
    output_path = 'challenging_defect_image.jpg'
    Image.fromarray(final_image).save(output_path)
    print(f"Saved defect image as: {output_path}")
    
    # 6. 創建標註圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左圖：原始圖片
    ax1.imshow(img_np, cmap='gray')
    ax1.set_title('Original Image', fontsize=14)
    ax1.axis('off')
    
    # 右圖：缺陷圖片with標註
    ax2.imshow(final_image, cmap='gray')
    ax2.set_title(f'Challenging Defect Image ({len(defect_info)} defects)', fontsize=14)
    
    # 標註缺陷位置
    for defect in defect_info:
        x, y, w, h = defect['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor='red', facecolor='none')
        ax2.add_patch(rect)
        
        # 添加標籤
        ax2.text(x + w/2, y - 5, f"D{defect['id']}", 
                color='red', fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('challenging_defect_annotated.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 7. 保存缺陷信息
    print("\nDefect Information:")
    print("=" * 50)
    for defect in defect_info:
        x, y, w, h = defect['bbox']
        print(f"Defect {defect['id']}:")
        print(f"  Location: ({x}, {y})")
        print(f"  Size: {w} x {h}")
        print(f"  Brightness: {defect['brightness']:.1f}")
        print(f"  Bbox: ({x}, {y}, {x+w}, {y+h})")
        print()
    
    # 保存缺陷信息到文件
    with open('defect_locations.txt', 'w') as f:
        f.write("Defect Locations for challenging_defect_image.jpg\n")
        f.write("=" * 50 + "\n\n")
        for defect in defect_info:
            x, y, w, h = defect['bbox']
            f.write(f"Defect {defect['id']}:\n")
            f.write(f"  Bbox (x, y, w, h): ({x}, {y}, {w}, {h})\n")
            f.write(f"  Top-left: ({x}, {y})\n")
            f.write(f"  Bottom-right: ({x+w}, {y+h})\n")
            f.write(f"  Center: {defect['center']}\n")
            f.write(f"  Brightness: {defect['brightness']:.1f}\n\n")
    
    print(f"\nDefect locations saved to: defect_locations.txt")
    print(f"Annotated image saved as: challenging_defect_annotated.png")


if __name__ == "__main__":
    create_challenging_defect_image()