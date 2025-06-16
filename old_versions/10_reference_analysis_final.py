import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from reference_finder import ReferenceFinder

# 初始化分析器
import os
if os.path.exists('Img_preprocess/image_with_bump.jpg'):
    image_path = 'Img_preprocess/image_with_bump.jpg'
else:
    image_path = 'image_with_bump.jpg'
finder = ReferenceFinder(image_path)
finder.detect_grid_lines()
finder.create_grid_cells()
finder.classify_cells()

# 使用者框選的缺陷區域
x1, y1 = 548, 494
x2, y2 = 560, 527
target_bbox = (x1, y1, x2-x1, y2-y1)

# 找出目標單元
target_cell = finder.find_target_cell(target_bbox)

# 找出參考單元
finder.find_reference_cells(target_cell, edge_margin=100, 
                           brightness_tolerance=10, size_tolerance=20)

# 創建詳細的視覺化
plt.ioff()
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# 顯示原圖
ax.imshow(finder.image_np, cmap='gray')
ax.set_title('Defect Detection Analysis - Complete View', fontsize=16, weight='bold')

# 繪製網格線
for v_line in finder.vertical_lines:
    ax.axvline(x=v_line, color='gray', linewidth=0.3, alpha=0.5)
for h_line in finder.horizontal_lines:
    ax.axhline(y=h_line, color='gray', linewidth=0.3, alpha=0.5)

# 標記不同排除原因的單元
for cell in finder.grid_cells:
    if cell['type'] != 'gray':
        continue
        
    # 判斷排除原因
    is_excluded = True
    color = None
    label = None
    
    if cell == target_cell:
        color = 'red'
        label = 'Target Cell'
        alpha = 0.2
    elif cell in finder.reference_cells:
        color = 'green'
        label = 'Selected'
        alpha = 0.2
        is_excluded = False
    elif cell['distance_to_edge'] < 100:
        color = 'yellow'
        label = f'Edge\n{cell["distance_to_edge"]}px'
        alpha = 0.3
    else:
        brightness_diff = abs(cell['mean_brightness'] - target_cell['mean_brightness'])
        if brightness_diff > 10:
            color = 'orange'
            label = f'Bright\n±{brightness_diff:.0f}'
            alpha = 0.3
    
    if color:
        rect = patches.Rectangle((cell['x1'], cell['y1']), 
                               cell['width'], cell['height'],
                               linewidth=1.5, edgecolor=color, 
                               facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        
        # 對右半邊的單元添加標籤
        if cell['x1'] > finder.width // 2 and is_excluded and cell != target_cell:
            ax.text(cell['x1'] + cell['width']//2, cell['y1'] + cell['height']//2, 
                   label, color='black', fontsize=7, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

# 重要：繪製目標物件框（紅色粗框）
x, y, w, h = target_bbox
target_rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                              edgecolor='red', facecolor='none', linestyle='-')
ax.add_patch(target_rect)

# 計算目標框的亮度
target_region = finder.image_np[y:y+h, x:x+w]
target_brightness = np.mean(target_region)

# 計算參考區域的亮度
reference_brightnesses = []
for ref_cell in finder.reference_cells:
    # 在參考cell中央取與目標框相同大小的區域
    cell_center_x = (ref_cell['x1'] + ref_cell['x2']) // 2
    cell_center_y = (ref_cell['y1'] + ref_cell['y2']) // 2
    
    ref_x = max(ref_cell['x1'], min(cell_center_x - w//2, ref_cell['x2'] - w))
    ref_y = max(ref_cell['y1'], min(cell_center_y - h//2, ref_cell['y2'] - h))
    
    ref_region = finder.image_np[ref_y:ref_y+h, ref_x:ref_x+w]
    reference_brightnesses.append(np.mean(ref_region))

ref_mean_brightness = np.mean(reference_brightnesses)
brightness_ratio = target_brightness / ref_mean_brightness

# 添加圖例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, alpha=0.5, label='Target cell'),
    Line2D([0], [0], color='red', linewidth=3, label='Target bbox (defect)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, alpha=0.5, label='Selected reference'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', markersize=10, alpha=0.5, label='Too close to edge'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10, alpha=0.5, label='Brightness mismatch'),
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

# 添加垂直線標記右半邊
ax.axvline(x=finder.width//2, color='blue', linewidth=2, linestyle='--', alpha=0.5)
ax.text(finder.width//2 + 10, 50, 'Right Half', color='blue', fontsize=12, weight='bold')

# 在右側添加統計信息框
info_text = f"=== ANALYSIS RESULTS ===\n\n"
info_text += f"Target Bbox (Red Box):\n"
info_text += f"  Position: ({x}, {y})\n"
info_text += f"  Size: {w} × {h} pixels\n"
info_text += f"  Brightness: {target_brightness:.2f}\n\n"

info_text += f"Reference Analysis:\n"
info_text += f"  Green boxes count: {len(finder.reference_cells)}\n"
info_text += f"  Avg brightness: {ref_mean_brightness:.2f}\n"
info_text += f"  Brightness range: {min(reference_brightnesses):.1f} - {max(reference_brightnesses):.1f}\n\n"

info_text += f"Signal Analysis:\n"
info_text += f"  Brightness Ratio: {brightness_ratio:.3f}×\n"
info_text += f"  Difference: +{target_brightness - ref_mean_brightness:.2f}\n\n"

if brightness_ratio > 1.3:
    info_text += f">> BRIGHT DEFECT DETECTED\n"
    defect_color = 'red'
elif brightness_ratio < 0.7:
    info_text += f">> DARK DEFECT DETECTED\n"
    defect_color = 'blue'
else:
    info_text += f">> NORMAL (No defect)\n"
    defect_color = 'green'

# 創建文字框
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
ax.text(1050, 400, info_text, fontsize=11, 
        bbox=props, verticalalignment='top', fontfamily='monospace')

# 標註目標框
ax.annotate('Target Defect', xy=(x+w/2, y+h), xytext=(x+w/2, y-30),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, color='red', weight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

plt.tight_layout()
plt.savefig('reference_analysis_final.png', dpi=150, bbox_inches='tight')
plt.close()

# 打印結果到終端
print("\n" + info_text)
print("Visualization saved as 'reference_analysis_final.png'")