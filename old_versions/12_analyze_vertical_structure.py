#!/usr/bin/env python3
"""
詳細分析垂直結構，找出正確的垂直邊緣位置
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import cv2

# 讀取圖片
img = Image.open('Img_preprocess/sem_noisy_output_raw.jpg').convert('L')
img_np = np.array(img)

# 計算列平均亮度
column_means = np.mean(img_np, axis=0)

# 計算梯度（一階導數）
gradient = np.gradient(column_means)

# 計算絕對梯度
abs_gradient = np.abs(gradient)

# 創建視覺化
fig, axes = plt.subplots(4, 1, figsize=(16, 12))

# 1. 原始圖片的一部分
ax1 = axes[0]
ax1.imshow(img_np[:200, :], cmap='gray', aspect='auto')
ax1.set_title('Original Image (top 200 rows)', fontsize=14)
ax1.set_xlabel('X (pixels)')

# 2. 列平均亮度
ax2 = axes[1]
ax2.plot(column_means, 'b-', linewidth=1)
ax2.set_title('Column Mean Brightness', fontsize=14)
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Brightness')
ax2.grid(True, alpha=0.3)

# 標記亮區和暗區
bright_threshold = np.percentile(column_means, 70)
dark_threshold = np.percentile(column_means, 30)
ax2.axhline(y=bright_threshold, color='r', linestyle='--', alpha=0.5, label=f'Bright threshold ({bright_threshold:.1f})')
ax2.axhline(y=dark_threshold, color='g', linestyle='--', alpha=0.5, label=f'Dark threshold ({dark_threshold:.1f})')
ax2.legend()

# 3. 梯度（邊緣強度）
ax3 = axes[2]
ax3.plot(gradient, 'r-', linewidth=1, label='Gradient')
ax3.set_title('Brightness Gradient (First Derivative)', fontsize=14)
ax3.set_xlabel('X (pixels)')
ax3.set_ylabel('Gradient')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# 4. 絕對梯度與邊緣檢測
ax4 = axes[3]
ax4.plot(abs_gradient, 'g-', linewidth=1)
ax4.set_title('Absolute Gradient with Edge Detection', fontsize=14)
ax4.set_xlabel('X (pixels)')
ax4.set_ylabel('|Gradient|')
ax4.grid(True, alpha=0.3)

# 找出梯度峰值（邊緣）
# 方法1：找正梯度峰值（從暗到亮）
pos_peaks, _ = find_peaks(gradient, height=np.std(gradient)*1.5, distance=30)
# 方法2：找負梯度峰值（從亮到暗）
neg_peaks, _ = find_peaks(-gradient, height=np.std(gradient)*1.5, distance=30)

print(f"Found {len(pos_peaks)} positive edges (dark to bright)")
print(f"Found {len(neg_peaks)} negative edges (bright to dark)")

# 在圖上標記邊緣
for peak in pos_peaks:
    ax4.axvline(x=peak, color='b', alpha=0.5, linewidth=2, label='Dark→Bright' if peak == pos_peaks[0] else '')
for peak in neg_peaks:
    ax4.axvline(x=peak, color='r', alpha=0.5, linewidth=2, label='Bright→Dark' if peak == neg_peaks[0] else '')

# 在原圖上標記
for peak in pos_peaks:
    ax1.axvline(x=peak, color='b', alpha=0.7, linewidth=1.5)
for peak in neg_peaks:
    ax1.axvline(x=peak, color='r', alpha=0.7, linewidth=1.5)

ax4.legend()

plt.tight_layout()
plt.savefig('vertical_structure_analysis.png', dpi=150)
plt.close()

# 分析週期性
all_edges = sorted(list(pos_peaks) + list(neg_peaks))
if len(all_edges) > 1:
    distances = np.diff(all_edges)
    print(f"\nEdge spacings: {distances}")
    print(f"Mean spacing: {np.mean(distances):.1f} pixels")
    print(f"Std spacing: {np.std(distances):.1f} pixels")

# 找出白條的位置
print("\n分析白條位置：")
in_bright = False
bright_regions = []
start = 0

for i, brightness in enumerate(column_means):
    if brightness > bright_threshold and not in_bright:
        in_bright = True
        start = i
    elif brightness < bright_threshold and in_bright:
        in_bright = False
        if i - start > 20:  # 至少20像素寬
            bright_regions.append((start, i))
            print(f"White strip: x={start}-{i}, width={i-start}")

# 使用 Canny 邊緣檢測作為對比
edges_canny = cv2.Canny(img_np, 50, 150)
vertical_projection = np.sum(edges_canny, axis=0)

# 創建第二個圖：比較不同方法
fig2, axes2 = plt.subplots(3, 1, figsize=(16, 10))

# 原圖
ax1 = axes2[0]
ax1.imshow(img_np[:200, :], cmap='gray', aspect='auto')
ax1.set_title('Comparison of Edge Detection Methods', fontsize=14)

# 我們的方法
for peak in pos_peaks:
    ax1.axvline(x=peak, color='b', alpha=0.7, linewidth=2, linestyle='-')
for peak in neg_peaks:
    ax1.axvline(x=peak, color='r', alpha=0.7, linewidth=2, linestyle='-')
ax1.text(10, 180, 'Blue: Dark→Bright edges, Red: Bright→Dark edges', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Canny 邊緣
ax2 = axes2[1]
ax2.imshow(edges_canny[:200, :], cmap='gray', aspect='auto')
ax2.set_title('Canny Edge Detection', fontsize=14)

# Canny 垂直投影
ax3 = axes2[2]
ax3.plot(vertical_projection)
ax3.set_title('Vertical Projection of Canny Edges', fontsize=14)
ax3.set_xlabel('X (pixels)')
ax3.grid(True, alpha=0.3)

# 找 Canny 的峰值
canny_peaks, _ = find_peaks(vertical_projection, height=np.max(vertical_projection)*0.3, distance=30)
for peak in canny_peaks:
    ax3.axvline(x=peak, color='g', alpha=0.5, linewidth=2)

plt.tight_layout()
plt.savefig('edge_detection_comparison.png', dpi=150)
plt.close()

# 輸出正確的垂直邊緣位置
print("\n建議的垂直邊緣位置（從暗到亮）：")
for i, edge in enumerate(pos_peaks):
    print(f"Edge {i+1}: x={edge}")

print("\n建議的垂直邊緣位置（從亮到暗）：")
for i, edge in enumerate(neg_peaks):
    print(f"Edge {i+1}: x={edge}")

# 創建一個更準確的邊緣列表
accurate_edges = []
# 對每個白條，取其左右邊緣
for start, end in bright_regions:
    # 左邊緣（暗到亮）
    left_edge = None
    for edge in pos_peaks:
        if abs(edge - start) < 10:
            left_edge = edge
            break
    if left_edge:
        accurate_edges.append(('left', left_edge))
    
    # 右邊緣（亮到暗）
    right_edge = None
    for edge in neg_peaks:
        if abs(edge - end) < 10:
            right_edge = edge
            break
    if right_edge:
        accurate_edges.append(('right', right_edge))

print("\n最終推薦的邊緣位置（用於放置缺陷）：")
for edge_type, pos in accurate_edges:
    print(f"{edge_type} edge at x={pos}")