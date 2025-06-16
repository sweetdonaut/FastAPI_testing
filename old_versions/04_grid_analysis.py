import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import signal

class GridAnalyzer:
    def __init__(self, image_path):
        """初始化分析器"""
        self.image = Image.open(image_path).convert('L')
        self.image_np = np.array(self.image)
        self.height, self.width = self.image_np.shape
        self.vertical_lines = []
        self.horizontal_lines = []
        self.grid_cells = []
        
    def detect_grid_lines(self):
        """檢測網格線（垂直和水平）"""
        # 檢測垂直線
        column_means = np.mean(self.image_np, axis=0)
        col_gradient = np.gradient(column_means)
        
        # 找出亮度變化大的位置（可能是邊界）
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
        
        print(f"Found {len(self.vertical_lines)} vertical lines")
        print(f"Found {len(self.horizontal_lines)} horizontal lines")
        
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
                
                # 計算單元的平均亮度
                cell_region = self.image_np[cell['y1']:cell['y2'], cell['x1']:cell['x2']]
                if cell_region.size > 0:
                    cell['mean_brightness'] = np.mean(cell_region)
                    cell['std_brightness'] = np.std(cell_region)
                else:
                    cell['mean_brightness'] = 0
                    cell['std_brightness'] = 0
                
                self.grid_cells.append(cell)
        
        return self.grid_cells
    
    def classify_cells(self):
        """分類網格單元（亮條、灰色方塊、暗區）"""
        brightnesses = [cell['mean_brightness'] for cell in self.grid_cells]
        
        # 使用聚類的方法分類
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
    
    def visualize_grid_analysis(self, target_bbox, save_path=None):
        """視覺化網格分析結果"""
        plt.ioff()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左圖：顯示原圖和網格線
        ax1.imshow(self.image_np, cmap='gray')
        ax1.set_title('Grid Structure Detection')
        
        # 繪製垂直線
        for v_line in self.vertical_lines:
            ax1.axvline(x=v_line, color='yellow', linewidth=0.5, alpha=0.7)
        
        # 繪製水平線
        for h_line in self.horizontal_lines:
            ax1.axhline(y=h_line, color='yellow', linewidth=0.5, alpha=0.7)
        
        # 繪製目標框
        x, y, w, h = target_bbox
        target_rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                      edgecolor='red', facecolor='none')
        ax1.add_patch(target_rect)
        
        # 找到並高亮目標單元
        target_cell = self.find_target_cell(target_bbox)
        if target_cell:
            cell_rect = patches.Rectangle((target_cell['x1'], target_cell['y1']), 
                                        target_cell['width'], target_cell['height'],
                                        linewidth=3, edgecolor='lime', 
                                        facecolor='lime', alpha=0.3)
            ax1.add_patch(cell_rect)
            ax1.text(target_cell['x1'] + 5, target_cell['y1'] + 15, 
                    'Target Cell', color='lime', fontsize=10, weight='bold')
        
        # 右圖：顯示分類後的網格
        ax2.imshow(self.image_np, cmap='gray', alpha=0.3)
        ax2.set_title('Cell Classification')
        
        # 根據類型著色網格單元
        for cell in self.grid_cells:
            if cell['type'] == 'bright':
                color = 'white'
                alpha = 0.3
            elif cell['type'] == 'dark':
                color = 'black'
                alpha = 0.3
            else:  # gray
                color = 'gray'
                alpha = 0.2
            
            rect = patches.Rectangle((cell['x1'], cell['y1']), 
                                   cell['width'], cell['height'],
                                   linewidth=0.5, edgecolor='black', 
                                   facecolor=color, alpha=alpha)
            ax2.add_patch(rect)
        
        # 再次高亮目標單元
        if target_cell:
            cell_rect = patches.Rectangle((target_cell['x1'], target_cell['y1']), 
                                        target_cell['width'], target_cell['height'],
                                        linewidth=3, edgecolor='red', 
                                        facecolor='none')
            ax2.add_patch(cell_rect)
        
        # 顯示目標框
        target_rect2 = patches.Rectangle((x, y), w, h, linewidth=2, 
                                       edgecolor='red', facecolor='none')
        ax2.add_patch(target_rect2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 打印分析結果
        print("\n=== Grid Analysis Results ===")
        print(f"Grid size: {len(self.horizontal_lines)+1} x {len(self.vertical_lines)+1}")
        print(f"Total cells: {len(self.grid_cells)}")
        
        # 統計各類型單元
        cell_types = {}
        for cell in self.grid_cells:
            cell_types[cell['type']] = cell_types.get(cell['type'], 0) + 1
        
        print("\nCell type distribution:")
        for cell_type, count in cell_types.items():
            print(f"  {cell_type}: {count} cells")
        
        if target_cell:
            print(f"\nTarget cell found:")
            print(f"  Position: Row {target_cell['row']}, Col {target_cell['col']}")
            print(f"  Bounds: ({target_cell['x1']}, {target_cell['y1']}) to "
                  f"({target_cell['x2']}, {target_cell['y2']})")
            print(f"  Size: {target_cell['width']} x {target_cell['height']}")
            print(f"  Type: {target_cell['type']}")
            print(f"  Mean brightness: {target_cell['mean_brightness']:.1f}")


# 測試程式碼
if __name__ == "__main__":
    # 初始化分析器
    analyzer = GridAnalyzer('image_with_bump.jpg')
    
    # 檢測網格線
    analyzer.detect_grid_lines()
    
    # 創建網格單元
    analyzer.create_grid_cells()
    
    # 分類單元
    analyzer.classify_cells()
    
    # 使用者框選的缺陷區域
    x1, y1 = 548, 494
    x2, y2 = 560, 527
    target_bbox = (x1, y1, x2-x1, y2-y1)
    
    # 視覺化結果
    analyzer.visualize_grid_analysis(target_bbox, save_path='grid_analysis_result.png')