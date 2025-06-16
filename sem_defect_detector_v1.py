#!/usr/bin/env python3
"""
SEM Defect Detector v1.0
掃描式電子顯微鏡影像缺陷檢測系統

功能：
1. 自動識別圖片網格結構
2. 智能選擇參考區域
3. 計算亮度比例判斷缺陷
4. 生成詳細的分析報告和視覺化
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import os
import sys


class SEMDefectDetector:
    """SEM 影像缺陷檢測器"""
    
    def __init__(self, image_path):
        """初始化檢測器"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        self.image = Image.open(image_path).convert('L')
        self.image_np = np.array(self.image)
        self.height, self.width = self.image_np.shape
        self.vertical_lines = []
        self.horizontal_lines = []
        self.grid_cells = []
        self.reference_cells = []
        self.excluded_cells = {}
        
    def detect_grid_lines(self):
        """檢測網格線（垂直和水平）"""
        # 檢測垂直線
        column_means = np.mean(self.image_np, axis=0)
        col_gradient = np.gradient(column_means)
        col_threshold = np.std(col_gradient) * 2
        vertical_edges = np.where(np.abs(col_gradient) > col_threshold)[0]
        
        # 合併相近的邊緣
        self.vertical_lines = []
        if len(vertical_edges) > 0:
            current_group = [vertical_edges[0]]
            for i in range(1, len(vertical_edges)):
                if vertical_edges[i] - vertical_edges[i-1] < 10:
                    current_group.append(vertical_edges[i])
                else:
                    self.vertical_lines.append(int(np.mean(current_group)))
                    current_group = [vertical_edges[i]]
            self.vertical_lines.append(int(np.mean(current_group)))
        
        # 檢測水平線
        row_means = np.mean(self.image_np, axis=1)
        row_gradient = np.gradient(row_means)
        row_threshold = np.std(row_gradient) * 2
        horizontal_edges = np.where(np.abs(row_gradient) > row_threshold)[0]
        
        # 合併相近的邊緣
        self.horizontal_lines = []
        if len(horizontal_edges) > 0:
            current_group = [horizontal_edges[0]]
            for i in range(1, len(horizontal_edges)):
                if horizontal_edges[i] - horizontal_edges[i-1] < 10:
                    current_group.append(horizontal_edges[i])
                else:
                    self.horizontal_lines.append(int(np.mean(current_group)))
                    current_group = [horizontal_edges[i]]
            self.horizontal_lines.append(int(np.mean(current_group)))
        
        return len(self.vertical_lines), len(self.horizontal_lines)
    
    def create_grid_cells(self):
        """根據檢測到的線條創建網格單元"""
        self.grid_cells = []
        
        # 添加邊界
        v_lines = [0] + sorted(self.vertical_lines) + [self.width]
        h_lines = [0] + sorted(self.horizontal_lines) + [self.height]
        
        # 創建所有網格單元
        for i in range(len(h_lines) - 1):
            for j in range(len(v_lines) - 1):
                cell = {
                    'row': i,
                    'col': j,
                    'x1': v_lines[j],
                    'y1': h_lines[i],
                    'x2': v_lines[j + 1],
                    'y2': h_lines[i + 1],
                    'width': v_lines[j + 1] - v_lines[j],
                    'height': h_lines[i + 1] - h_lines[i]
                }
                
                # 計算單元的平均亮度和標準差
                cell_region = self.image_np[cell['y1']:cell['y2'], cell['x1']:cell['x2']]
                if cell_region.size > 0:
                    cell['mean_brightness'] = np.mean(cell_region)
                    cell['std_brightness'] = np.std(cell_region)
                else:
                    cell['mean_brightness'] = 0
                    cell['std_brightness'] = 0
                
                # 計算到邊界的距離
                cell['distance_to_edge'] = min(
                    cell['x1'],  # 左邊界
                    cell['y1'],  # 上邊界
                    self.width - cell['x2'],  # 右邊界
                    self.height - cell['y2']  # 下邊界
                )
                
                self.grid_cells.append(cell)
        
        return len(self.grid_cells)
    
    def classify_cells(self):
        """分類網格單元（亮條、灰色方塊、暗區）"""
        brightnesses = [cell['mean_brightness'] for cell in self.grid_cells]
        bright_threshold = np.percentile(brightnesses, 75)
        dark_threshold = np.percentile(brightnesses, 25)
        
        for cell in self.grid_cells:
            if cell['mean_brightness'] > bright_threshold:
                cell['type'] = 'bright'
            elif cell['mean_brightness'] < dark_threshold:
                cell['type'] = 'dark'
            else:
                cell['type'] = 'gray'
        
        # 統計各類型數量
        type_counts = {'bright': 0, 'gray': 0, 'dark': 0}
        for cell in self.grid_cells:
            type_counts[cell['type']] += 1
            
        return type_counts
    
    def find_target_cell(self, target_bbox):
        """找出目標框所在的網格單元"""
        x, y, w, h = target_bbox
        target_center_x = x + w // 2
        target_center_y = y + h // 2
        
        target_cell = None
        for cell in self.grid_cells:
            if (cell['x1'] <= target_center_x <= cell['x2'] and 
                cell['y1'] <= target_center_y <= cell['y2']):
                target_cell = cell
                break
        
        return target_cell
    
    def find_reference_cells(self, target_cell, edge_margin=100, 
                           brightness_tolerance=10, size_tolerance=20):
        """根據目標單元找出合適的參考單元"""
        self.reference_cells = []
        self.excluded_cells = {
            'edge': [],
            'brightness': [],
            'size': [],
            'type': []
        }
        
        if not target_cell:
            return []
        
        target_brightness = target_cell['mean_brightness']
        target_width = target_cell['width']
        target_height = target_cell['height']
        
        for cell in self.grid_cells:
            # 排除目標單元本身
            if cell == target_cell:
                continue
            
            # 排除太靠近邊界的單元
            if cell['distance_to_edge'] < edge_margin:
                self.excluded_cells['edge'].append(cell)
                continue
            
            # 只選擇灰色方塊
            if cell['type'] != 'gray':
                self.excluded_cells['type'].append(cell)
                continue
            
            # 檢查亮度相似性
            brightness_diff = abs(cell['mean_brightness'] - target_brightness)
            if brightness_diff > brightness_tolerance:
                self.excluded_cells['brightness'].append(cell)
                continue
            
            # 檢查尺寸相似性
            width_diff = abs(cell['width'] - target_width)
            height_diff = abs(cell['height'] - target_height)
            if width_diff > size_tolerance or height_diff > size_tolerance:
                self.excluded_cells['size'].append(cell)
                continue
            
            # 計算綜合相似度分數
            brightness_score = 1 - (brightness_diff / brightness_tolerance)
            size_score = 1 - (width_diff + height_diff) / (2 * size_tolerance)
            distance_score = cell['distance_to_edge'] / self.width
            
            cell['similarity_score'] = (brightness_score + size_score + distance_score) / 3
            self.reference_cells.append(cell)
        
        # 按相似度分數排序
        self.reference_cells.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return len(self.reference_cells)
    
    def analyze_defect(self, target_bbox, edge_margin=100, 
                      brightness_tolerance=10, size_tolerance=20):
        """執行完整的缺陷分析"""
        # 檢測網格
        v_count, h_count = self.detect_grid_lines()
        print(f"Grid detected: {v_count} vertical lines, {h_count} horizontal lines")
        
        # 創建網格單元
        cell_count = self.create_grid_cells()
        print(f"Created {cell_count} grid cells")
        
        # 分類單元
        type_counts = self.classify_cells()
        print(f"Cell types: {type_counts}")
        
        # 找目標單元
        target_cell = self.find_target_cell(target_bbox)
        if not target_cell:
            raise ValueError("Target cell not found!")
        print(f"Target cell found at Row {target_cell['row']}, Col {target_cell['col']}")
        
        # 找參考單元
        ref_count = self.find_reference_cells(target_cell, edge_margin, 
                                             brightness_tolerance, size_tolerance)
        print(f"Found {ref_count} reference cells")
        
        # 計算目標框的亮度
        x, y, w, h = target_bbox
        target_region = self.image_np[y:y+h, x:x+w]
        target_brightness = np.mean(target_region)
        target_std = np.std(target_region)
        
        # 計算參考區域的亮度
        reference_brightnesses = []
        for ref_cell in self.reference_cells:
            # 在參考cell中央取與目標框相同大小的區域
            cell_center_x = (ref_cell['x1'] + ref_cell['x2']) // 2
            cell_center_y = (ref_cell['y1'] + ref_cell['y2']) // 2
            
            ref_x = max(ref_cell['x1'], min(cell_center_x - w//2, ref_cell['x2'] - w))
            ref_y = max(ref_cell['y1'], min(cell_center_y - h//2, ref_cell['y2'] - h))
            
            ref_region = self.image_np[ref_y:ref_y+h, ref_x:ref_x+w]
            reference_brightnesses.append(np.mean(ref_region))
        
        if not reference_brightnesses:
            raise ValueError("No reference regions found!")
        
        ref_mean = np.mean(reference_brightnesses)
        ref_std = np.std(reference_brightnesses)
        brightness_ratio = target_brightness / ref_mean
        
        # 判斷缺陷類型
        if brightness_ratio > 1.3:
            defect_type = "BRIGHT DEFECT"
        elif brightness_ratio < 0.7:
            defect_type = "DARK DEFECT"
        else:
            defect_type = "NORMAL"
        
        results = {
            'target_bbox': target_bbox,
            'target_brightness': target_brightness,
            'target_std': target_std,
            'reference_count': len(reference_brightnesses),
            'reference_mean': ref_mean,
            'reference_std': ref_std,
            'reference_range': (min(reference_brightnesses), max(reference_brightnesses)),
            'brightness_ratio': brightness_ratio,
            'defect_type': defect_type,
            'target_cell': target_cell
        }
        
        return results
    
    def visualize_results(self, target_bbox, results, save_path='defect_analysis_result.png'):
        """視覺化分析結果"""
        plt.ioff()
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # 顯示原圖
        ax.imshow(self.image_np, cmap='gray')
        ax.set_title('SEM Defect Detection Analysis v1.0', fontsize=16, weight='bold')
        
        # 繪製網格線
        for v_line in self.vertical_lines:
            ax.axvline(x=v_line, color='gray', linewidth=0.3, alpha=0.5)
        for h_line in self.horizontal_lines:
            ax.axhline(y=h_line, color='gray', linewidth=0.3, alpha=0.5)
        
        # 標記不同排除原因的單元
        target_cell = results['target_cell']
        
        for cell in self.grid_cells:
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
            elif cell in self.reference_cells:
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
                if cell['x1'] > self.width // 2 and is_excluded and cell != target_cell:
                    ax.text(cell['x1'] + cell['width']//2, cell['y1'] + cell['height']//2, 
                           label, color='black', fontsize=7, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # 繪製目標物件框（紅色粗框）
        x, y, w, h = target_bbox
        target_rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                      edgecolor='red', facecolor='none', linestyle='-')
        ax.add_patch(target_rect)
        
        # 添加圖例
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                   markersize=10, alpha=0.5, label='Target cell'),
            Line2D([0], [0], color='red', linewidth=3, label='Target bbox (defect)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                   markersize=10, alpha=0.5, label='Selected reference'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', 
                   markersize=10, alpha=0.5, label='Too close to edge'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', 
                   markersize=10, alpha=0.5, label='Brightness mismatch'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # 添加垂直線標記右半邊
        ax.axvline(x=self.width//2, color='blue', linewidth=2, linestyle='--', alpha=0.5)
        ax.text(self.width//2 + 10, 50, 'Right Half', color='blue', fontsize=12, weight='bold')
        
        # 在右側添加統計信息框
        info_text = self._format_results(results)
        
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
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved as '{save_path}'")
    
    def _format_results(self, results):
        """格式化結果文字"""
        x, y, w, h = results['target_bbox']
        
        info_text = f"=== ANALYSIS RESULTS ===\n\n"
        info_text += f"Target Bbox (Red Box):\n"
        info_text += f"  Position: ({x}, {y})\n"
        info_text += f"  Size: {w} × {h} pixels\n"
        info_text += f"  Brightness: {results['target_brightness']:.2f}\n\n"
        
        info_text += f"Reference Analysis:\n"
        info_text += f"  Green boxes count: {results['reference_count']}\n"
        info_text += f"  Avg brightness: {results['reference_mean']:.2f}\n"
        info_text += f"  Brightness range: {results['reference_range'][0]:.1f} - {results['reference_range'][1]:.1f}\n\n"
        
        info_text += f"Signal Analysis:\n"
        info_text += f"  Brightness Ratio: {results['brightness_ratio']:.3f}×\n"
        info_text += f"  Difference: +{results['target_brightness'] - results['reference_mean']:.2f}\n\n"
        
        if results['defect_type'] == 'BRIGHT DEFECT':
            info_text += f">> BRIGHT DEFECT DETECTED\n"
        elif results['defect_type'] == 'DARK DEFECT':
            info_text += f">> DARK DEFECT DETECTED\n"
        else:
            info_text += f">> NORMAL (No defect)\n"
        
        return info_text


def main():
    """主程式"""
    # 檢查圖片路徑
    if os.path.exists('Img_preprocess/image_with_bump.jpg'):
        image_path = 'Img_preprocess/image_with_bump.jpg'
    elif os.path.exists('image_with_bump.jpg'):
        image_path = 'image_with_bump.jpg'
    else:
        print("Error: Image file 'image_with_bump.jpg' not found!")
        sys.exit(1)
    
    print(f"Using image: {image_path}")
    
    try:
        # 初始化檢測器
        detector = SEMDefectDetector(image_path)
        
        # 定義目標缺陷區域
        x1, y1 = 548, 494
        x2, y2 = 560, 527
        target_bbox = (x1, y1, x2-x1, y2-y1)
        
        print(f"Target defect region: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"Target bbox: x={x1}, y={y1}, w={x2-x1}, h={y2-y1}\n")
        
        # 執行分析
        results = detector.analyze_defect(target_bbox, 
                                        edge_margin=100,
                                        brightness_tolerance=10, 
                                        size_tolerance=20)
        
        # 打印結果
        print("\n" + detector._format_results(results))
        
        # 生成視覺化
        detector.visualize_results(target_bbox, results, 
                                 save_path='sem_defect_analysis_v1.png')
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()