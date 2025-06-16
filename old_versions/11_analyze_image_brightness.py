#!/usr/bin/env python3
"""
分析圖片的亮度分布，找出為什麼檢測不到水平條帶
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 讀取圖片
img = Image.open('Img_preprocess/sem_noisy_output_raw.jpg').convert('L')
img_np = np.array(img)

# 計算每行的平均亮度
row_means = np.mean(img_np, axis=1)

# 創建視覺化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# 左圖：原始圖片
ax1.imshow(img_np, cmap='gray')
ax1.set_title('Original Image')
ax1.set_ylabel('Y (pixels)')

# 右圖：行亮度分布
ax2.plot(row_means, range(len(row_means)))
ax2.invert_yaxis()  # 翻轉y軸以匹配圖片
ax2.set_xlabel('Mean Brightness')
ax2.set_ylabel('Y (pixels)')
ax2.set_title('Row Brightness Profile')
ax2.grid(True, alpha=0.3)

# 標記一些閾值
percentiles = [50, 60, 65, 70, 75, 80]
for p in percentiles:
    val = np.percentile(row_means, p)
    ax2.axvline(x=val, color='red', alpha=0.3, linestyle='--', label=f'{p}%: {val:.1f}')

ax2.legend()

# 找出局部峰值
from scipy.signal import find_peaks

# 找峰值（亮條）
peaks, properties = find_peaks(row_means, height=np.percentile(row_means, 60), 
                              distance=20, width=5)

print(f"Image shape: {img_np.shape}")
print(f"Brightness range: {np.min(row_means):.1f} - {np.max(row_means):.1f}")
print(f"Mean brightness: {np.mean(row_means):.1f}")
print(f"Std brightness: {np.std(row_means):.1f}")
print(f"\nPercentiles:")
for p in percentiles:
    print(f"  {p}%: {np.percentile(row_means, p):.1f}")

print(f"\nFound {len(peaks)} peaks (potential bright bands)")
for i, peak in enumerate(peaks):
    print(f"  Peak {i+1}: y={peak}, brightness={row_means[peak]:.1f}")

# 在左圖標記峰值位置
for peak in peaks:
    ax1.axhline(y=peak, color='yellow', alpha=0.5, linewidth=2)

# 在右圖標記峰值
ax2.plot(row_means[peaks], peaks, 'ro', markersize=8, label='Peaks')

plt.tight_layout()
plt.savefig('brightness_analysis.png', dpi=150)
plt.close()

# 分析為什麼原方法失敗
print("\n分析原方法失敗原因：")
print("1. 圖片可能沒有明顯的水平亮條")
print("2. 亮條可能太窄或對比度太低")
print("3. 可能需要使用不同的檢測策略")

# 嘗試另一種方法：檢測暗條之間的區域
dark_threshold = np.percentile(row_means, 30)
bright_threshold = np.percentile(row_means, 70)

in_bright = False
bright_regions = []
start = 0

for i, val in enumerate(row_means):
    if val > bright_threshold and not in_bright:
        in_bright = True
        start = i
    elif val < dark_threshold and in_bright:
        in_bright = False
        if i - start > 10:  # 至少10像素寬
            bright_regions.append((start, i))

print(f"\n使用暗條分隔法找到 {len(bright_regions)} 個亮區域")
for i, (start, end) in enumerate(bright_regions):
    print(f"  Region {i+1}: y={start}-{end}, width={end-start}")

# 直接使用週期性模式
# SEM圖片通常有規律的條紋
print("\n嘗試找出週期性模式...")
# 計算自相關
from scipy import signal
autocorr = signal.correlate(row_means - np.mean(row_means), 
                           row_means - np.mean(row_means), mode='same')
autocorr = autocorr / np.max(autocorr)

# 找週期
peaks_auto, _ = find_peaks(autocorr[len(autocorr)//2:], height=0.3, distance=20)
if len(peaks_auto) > 0:
    period = peaks_auto[0]
    print(f"可能的週期: {period} 像素")
    
    # 基於週期創建條帶
    periodic_bands = []
    for i in range(0, img_np.shape[0], period):
        if i + 20 < img_np.shape[0]:
            periodic_bands.append((i, min(i + 30, img_np.shape[0])))
    
    print(f"基於週期創建了 {len(periodic_bands)} 個條帶")