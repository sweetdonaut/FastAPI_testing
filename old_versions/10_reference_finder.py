import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import signal

class ReferenceFinder:
    def __init__(self, image_path):
        """初始化分析器"""
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
        
        return self.vertical_lines, self.horizontal_lines
    
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
        
        return self.grid_cells
    
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
        
        return self.grid_cells
    
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
            distance_score = cell['distance_to_edge'] / self.width  # 離邊界越遠越好
            
            cell['similarity_score'] = (brightness_score + size_score + distance_score) / 3
            self.reference_cells.append(cell)
        
        # 按相似度分數排序
        self.reference_cells.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return self.reference_cells
    
    def visualize_references(self, target_bbox, max_refs=20, save_path=None, 
                           edge_margin=100, brightness_tolerance=10, size_tolerance=20):
        """視覺化參考區域選擇結果"""
        plt.ioff()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        
        # 左圖：顯示所有候選參考區域
        ax1.imshow(self.image_np, cmap='gray')
        ax1.set_title('Reference Cell Selection (Green: Selected, Yellow: Edge cells excluded)')
        
        # 繪製網格線
        for v_line in self.vertical_lines:
            ax1.axvline(x=v_line, color='gray', linewidth=0.3, alpha=0.5)
        for h_line in self.horizontal_lines:
            ax1.axhline(y=h_line, color='gray', linewidth=0.3, alpha=0.5)
        
        # 標記邊界區域（黃色）
        for cell in self.grid_cells:
            if cell['distance_to_edge'] < edge_margin and cell['type'] == 'gray':
                rect = patches.Rectangle((cell['x1'], cell['y1']), 
                                       cell['width'], cell['height'],
                                       linewidth=1, edgecolor='yellow', 
                                       facecolor='yellow', alpha=0.2)
                ax1.add_patch(rect)
        
        # 繪製目標框和單元
        x, y, w, h = target_bbox
        target_rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                      edgecolor='red', facecolor='none')
        ax1.add_patch(target_rect)
        
        target_cell = self.find_target_cell(target_bbox)
        if target_cell:
            target_cell_rect = patches.Rectangle((target_cell['x1'], target_cell['y1']), 
                                               target_cell['width'], target_cell['height'],
                                               linewidth=2, edgecolor='red', 
                                               facecolor='red', alpha=0.2)
            ax1.add_patch(target_cell_rect)
        
        # 繪製參考單元（綠色，最多顯示max_refs個）
        for i, cell in enumerate(self.reference_cells[:max_refs]):
            rect = patches.Rectangle((cell['x1'], cell['y1']), 
                                   cell['width'], cell['height'],
                                   linewidth=2, edgecolor='green', 
                                   facecolor='green', alpha=0.2)
            ax1.add_patch(rect)
            # 標註編號
            ax1.text(cell['x1'] + cell['width']//2, cell['y1'] + cell['height']//2, 
                    f'{i+1}', color='white', fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle="circle,pad=0.3", facecolor="green", alpha=0.8))
        
        # 右圖：顯示詳細信息
        ax2.axis('off')
        
        info_text = "=== Reference Selection Results ===\n\n"
        
        if target_cell:
            info_text += f"Target Cell:\n"
            info_text += f"  Position: Row {target_cell['row']}, Col {target_cell['col']}\n"
            info_text += f"  Brightness: {target_cell['mean_brightness']:.1f} ± {target_cell['std_brightness']:.1f}\n"
            info_text += f"  Size: {target_cell['width']}x{target_cell['height']}\n\n"
        
        info_text += f"Selection Criteria:\n"
        info_text += f"  Edge margin: >{edge_margin}px\n"
        info_text += f"  Brightness tolerance: ±{brightness_tolerance}\n"
        info_text += f"  Size tolerance: ±{size_tolerance}px\n\n"
        
        info_text += f"Exclusion Summary:\n"
        info_text += f"  Too close to edge: {len(self.excluded_cells['edge'])} cells\n"
        info_text += f"  Not gray cells: {len(self.excluded_cells['type'])} cells\n"
        info_text += f"  Brightness mismatch: {len(self.excluded_cells['brightness'])} cells\n"
        info_text += f"  Size mismatch: {len(self.excluded_cells['size'])} cells\n\n"
        
        info_text += f"Found {len(self.reference_cells)} suitable reference cells\n\n"
        
        if self.reference_cells:
            info_text += "Top 10 Reference Cells:\n"
            for i, cell in enumerate(self.reference_cells[:10]):
                info_text += f"\n{i+1}. Row {cell['row']}, Col {cell['col']}\n"
                info_text += f"   Brightness: {cell['mean_brightness']:.1f}\n"
                info_text += f"   Size: {cell['width']}x{cell['height']}\n"
                info_text += f"   Edge distance: {cell['distance_to_edge']}px\n"
                info_text += f"   Similarity score: {cell['similarity_score']:.3f}\n"
        
        # 計算參考單元的統計信息
        if self.reference_cells:
            ref_brightnesses = [cell['mean_brightness'] for cell in self.reference_cells]
            info_text += f"\nReference cells brightness:\n"
            info_text += f"  Mean: {np.mean(ref_brightnesses):.2f}\n"
            info_text += f"  Std: {np.std(ref_brightnesses):.2f}\n"
            info_text += f"  Range: {min(ref_brightnesses):.1f} - {max(ref_brightnesses):.1f}\n"
        
        ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(info_text)


# 測試程式碼
if __name__ == "__main__":
    # 初始化分析器
    finder = ReferenceFinder('image_with_bump.jpg')
    
    # 檢測網格
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
    
    # 視覺化結果
    finder.visualize_references(target_bbox, max_refs=20, 
                               save_path='reference_selection_result.png',
                               edge_margin=100, brightness_tolerance=10, size_tolerance=20)
    
    # 打印右半邊的分析
    print("\n=== Analysis of Right Half Cells ===")
    right_half_cells = [cell for cell in finder.grid_cells if cell['x1'] > finder.width // 2]
    gray_right_cells = [cell for cell in right_half_cells if cell['type'] == 'gray']
    
    print(f"Total cells in right half: {len(right_half_cells)}")
    print(f"Gray cells in right half: {len(gray_right_cells)}")
    
    # 分析為什麼右半邊的灰色方塊沒被選上
    for cell in gray_right_cells:
        if cell in finder.reference_cells:
            continue
            
        reasons = []
        if cell == target_cell:
            reasons.append("Is target cell")
        elif cell['distance_to_edge'] < 100:
            reasons.append(f"Too close to edge ({cell['distance_to_edge']}px)")
        else:
            brightness_diff = abs(cell['mean_brightness'] - target_cell['mean_brightness'])
            if brightness_diff > 10:
                reasons.append(f"Brightness diff too large ({brightness_diff:.1f})")
            
            width_diff = abs(cell['width'] - target_cell['width'])
            height_diff = abs(cell['height'] - target_cell['height'])
            if width_diff > 20 or height_diff > 20:
                reasons.append(f"Size diff too large (w:{width_diff}, h:{height_diff})")
        
        if reasons:
            print(f"  Row {cell['row']}, Col {cell['col']}: {'; '.join(reasons)}")