import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from reference_finder import ReferenceFinder

# 初始化分析器
finder = ReferenceFinder('image_with_bump.jpg')
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
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# 顯示原圖
ax.imshow(finder.image_np, cmap='gray')
ax.set_title('Detailed Analysis of Reference Selection\n(Right half exclusion reasons)')

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
        label = 'Target'
        alpha = 0.3
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

# 添加圖例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, alpha=0.5, label='Target cell'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, alpha=0.5, label='Selected reference'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', markersize=10, alpha=0.5, label='Too close to edge'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10, alpha=0.5, label='Brightness mismatch'),
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

# 添加垂直線標記右半邊
ax.axvline(x=finder.width//2, color='blue', linewidth=2, linestyle='--', alpha=0.5)
ax.text(finder.width//2 + 10, 50, 'Right Half', color='blue', fontsize=12, weight='bold')

plt.tight_layout()
plt.savefig('reference_analysis_detailed.png', dpi=150, bbox_inches='tight')
plt.close()

print("Detailed visualization saved as 'reference_analysis_detailed.png'")