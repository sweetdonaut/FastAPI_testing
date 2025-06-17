#!/usr/bin/env python3
"""
SEM 缺陷檢測器 v2.0
主要改進：更準確的單元格分類，避免選擇白色垂直條作為參考區域
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
import sys


class SEMDefectDetector:
    def __init__(self, image_path):
        """初始化檢測器"""
        self.image = Image.open(image_path).convert('L')
        self.image_np = np.array(self.image)
        self.height, self.width = self.image_np.shape
        
        # 儲存分析結果
        self.vertical_lines = []
        self.horizontal_lines = []
        self.grid_cells = []
        self.reference_cells = []
        self.excluded_cells = {'edge': [], 'brightness': [], 'size': [], 'white_stripe': []}
        
    def detect_grid_lines(self):
        """檢測網格線"""
        # 垂直線檢測
        v_projection = np.mean(self.image_np, axis=0)
        v_gradient = np.abs(np.diff(v_projection))
        v_threshold = np.percentile(v_gradient, 85)
        
        self.vertical_lines = []
        for i in range(1, len(v_gradient)-1):
            if v_gradient[i] > v_threshold:
                if not self.vertical_lines or i - self.vertical_lines[-1] > 20:
                    self.vertical_lines.append(i)
        
        # 水平線檢測
        h_projection = np.mean(self.image_np, axis=1)
        h_gradient = np.abs(np.diff(h_projection))
        h_threshold = np.percentile(h_gradient, 85)
        
        self.horizontal_lines = []
        for i in range(1, len(h_gradient)-1):
            if h_gradient[i] > h_threshold:
                if not self.horizontal_lines or i - self.horizontal_lines[-1] > 20:
                    self.horizontal_lines.append(i)
        
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
        """分類網格單元（白色垂直條、灰色方塊、暗區）"""
        # 改進的分類策略
        brightnesses = [cell['mean_brightness'] for cell in self.grid_cells]
        
        # 動態計算閾值
        # 使用K-means或閾值分析來找到三個主要群組
        sorted_brightness = sorted(brightnesses)
        n = len(sorted_brightness)
        
        # 找到三個主要亮度群組
        # 暗區：底部25%
        # 灰區：中間50%  
        # 亮區/白條：頂部25%
        dark_threshold = sorted_brightness[int(n * 0.3)]
        bright_threshold = sorted_brightness[int(n * 0.7)]
        
        print(f"Brightness thresholds - Dark: <{dark_threshold:.1f}, Bright: >{bright_threshold:.1f}")
        
        for cell in self.grid_cells:
            # 檢查是否為白色垂直條
            # 白色垂直條的特徵：非常高的亮度（>180）或高亮度+狹窄寬度
            if (cell['mean_brightness'] > 180 or 
                (cell['mean_brightness'] > bright_threshold and cell['width'] < 40)):
                cell['type'] = 'white_stripe'
            elif cell['mean_brightness'] < dark_threshold:
                cell['type'] = 'dark'
            else:
                cell['type'] = 'gray'
        
        # 統計各類型數量
        type_counts = {'white_stripe': 0, 'gray': 0, 'dark': 0}
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
        """尋找適合的參考單元"""
        self.reference_cells = []
        self.excluded_cells = {'edge': [], 'brightness': [], 'size': [], 'white_stripe': []}
        
        for cell in self.grid_cells:
            if cell == target_cell:
                continue
            
            # 排除白色垂直條
            if cell['type'] == 'white_stripe':
                self.excluded_cells['white_stripe'].append(cell)
                continue
            
            # 只考慮與目標相同類型的單元（如果目標不是白色條）
            if target_cell['type'] != 'white_stripe' and cell['type'] != target_cell['type']:
                continue
            
            # 檢查邊界距離
            if cell['distance_to_edge'] < edge_margin:
                self.excluded_cells['edge'].append(cell)
                continue
            
            # 檢查亮度差異
            brightness_diff = abs(cell['mean_brightness'] - target_cell['mean_brightness'])
            if brightness_diff > brightness_tolerance:
                self.excluded_cells['brightness'].append(cell)
                continue
            
            # 檢查尺寸差異
            width_diff = abs(cell['width'] - target_cell['width'])
            height_diff = abs(cell['height'] - target_cell['height'])
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
        print(f"Target cell type: {target_cell['type']}")
        
        # 找參考單元
        ref_count = self.find_reference_cells(target_cell, edge_margin, 
                                             brightness_tolerance, size_tolerance)
        print(f"Found {ref_count} reference cells")
        print(f"Excluded cells - White stripes: {len(self.excluded_cells['white_stripe'])}")
        
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
        ax.set_title('SEM Defect Detection Analysis v2.0', fontsize=16, weight='bold')
        
        # 繪製網格線
        for v_line in self.vertical_lines:
            ax.axvline(x=v_line, color='gray', linewidth=0.3, alpha=0.5)
        for h_line in self.horizontal_lines:
            ax.axhline(y=h_line, color='gray', linewidth=0.3, alpha=0.5)
        
        # 標記不同排除原因的單元
        target_cell = results['target_cell']
        
        for cell in self.grid_cells:
            # 跳過非灰色單元（除了白色垂直條）
            if cell['type'] not in ['gray', 'white_stripe']:
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
            elif cell in self.excluded_cells['white_stripe']:
                color = 'purple'
                label = 'White\nStripe'
                alpha = 0.3
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
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        # 標記目標框
        x, y, w, h = target_bbox
        target_rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                      edgecolor='red', facecolor='none')
        ax.add_patch(target_rect)
        ax.text(x-5, y-5, 'Target Defect', color='red', fontsize=10, 
               weight='bold', ha='right', va='bottom')
        
        # 標記參考區域（只標記前10個）
        for i, ref_cell in enumerate(self.reference_cells[:10]):
            cell_center_x = (ref_cell['x1'] + ref_cell['x2']) // 2
            cell_center_y = (ref_cell['y1'] + ref_cell['y2']) // 2
            
            ref_x = max(ref_cell['x1'], min(cell_center_x - w//2, ref_cell['x2'] - w))
            ref_y = max(ref_cell['y1'], min(cell_center_y - h//2, ref_cell['y2'] - h))
            
            ref_rect = patches.Rectangle((ref_x, ref_y), w, h, linewidth=1.5, 
                                       edgecolor='green', facecolor='none', linestyle='--')
            ax.add_patch(ref_rect)
            
            # 標記 REF
            ax.text(ref_x + w//2, ref_y + h//2, 'REF', color='green', 
                   fontsize=8, ha='center', va='center', weight='bold')
        
        # 添加分析結果文字框
        results_text = f"=== ANALYSIS RESULTS ===\n"
        results_text += f"\nTarget Bbox: {results['target_bbox']}"
        results_text += f"\nPosition: ({target_cell['row']}, {target_cell['col']})"
        results_text += f"\nCell Type: {target_cell['type']}"
        results_text += f"\nBrightness: {results['target_brightness']:.2f}"
        results_text += f"\n\nReference Analysis:"
        results_text += f"\n  Count: {results['reference_count']}"
        results_text += f"\n  Mean Brightness: {results['reference_mean']:.2f}"
        results_text += f"\n  Brightness Range: {results['reference_range'][0]:.1f} - {results['reference_range'][1]:.1f}"
        results_text += f"\n\nBrightness Ratio: {results['brightness_ratio']:.3f}×"
        results_text += f"\n\n>>> {results['defect_type']} <<<"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        ax.text(0.98, 0.5, results_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', ha='right', bbox=props, family='monospace')
        
        # 添加圖例
        legend_elements = [
            patches.Patch(color='red', alpha=0.5, label='Target Cell'),
            patches.Patch(color='green', alpha=0.5, label='Selected Reference'),
            patches.Patch(color='purple', alpha=0.5, label='White Stripe (Excluded)'),
            patches.Patch(color='yellow', alpha=0.5, label='Too close to edge'),
            patches.Patch(color='orange', alpha=0.5, label='Brightness mismatch'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved as '{save_path}'")
    
    def _format_results(self, results):
        """格式化結果輸出"""
        output = []
        output.append("=== DEFECT ANALYSIS RESULTS ===")
        output.append(f"Target brightness: {results['target_brightness']:.2f}")
        output.append(f"Reference mean: {results['reference_mean']:.2f}")
        output.append(f"Reference count: {results['reference_count']}")
        output.append(f"Brightness ratio: {results['brightness_ratio']:.3f}×")
        output.append(f"Detection result: {results['defect_type']}")
        return '\n'.join(output)


def main():
    """主程式 - 測試缺陷檢測"""
    if len(sys.argv) < 2:
        # 使用預設參數
        image_path = 'final_blurred_defects.jpg'
        target_bbox = (1006, 482, 11, 17)  # defect_6 的位置
    else:
        image_path = sys.argv[1]
        if len(sys.argv) >= 6:
            target_bbox = tuple(map(int, sys.argv[2:6]))
        else:
            print("Usage: python sem_defect_detector_v2.py <image_path> [x y w h]")
            sys.exit(1)
    
    print(f"Loading image: {image_path}")
    print(f"Target bbox: {target_bbox}")
    
    try:
        # 初始化檢測器
        detector = SEMDefectDetector(image_path)
        
        # 執行分析
        results = detector.analyze_defect(target_bbox, 
                                        edge_margin=100,
                                        brightness_tolerance=10, 
                                        size_tolerance=20)
        
        # 打印結果
        print("\n" + detector._format_results(results))
        
        # 生成視覺化
        detector.visualize_results(target_bbox, results, 
                                 save_path='sem_defect_analysis_v2.png')
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()