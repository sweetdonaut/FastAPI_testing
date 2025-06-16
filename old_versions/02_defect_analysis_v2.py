import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 預設字體
plt.rcParams['axes.unicode_minus'] = False

class DefectAnalyzer:
    def __init__(self, image_path):
        """初始化分析器"""
        self.image = Image.open(image_path).convert('L')
        self.image_np = np.array(self.image)
        self.height, self.width = self.image_np.shape
        self.vertical_lines_mask = None
        self.horizontal_bands = []
        self.reference_regions = []
        
    def detect_vertical_lines(self, threshold_factor=1.2):
        """檢測垂直亮條的位置"""
        # 計算每列的平均亮度
        column_means = np.mean(self.image_np, axis=0)
        
        # 使用閾值找出亮條
        mean_brightness = np.mean(column_means)
        threshold = mean_brightness * threshold_factor
        
        # 創建垂直線條的遮罩
        self.vertical_lines_mask = np.zeros_like(self.image_np, dtype=bool)
        for col in range(self.width):
            if column_means[col] > threshold:
                self.vertical_lines_mask[:, col] = True
                
        return self.vertical_lines_mask
    
    def detect_horizontal_bands(self, min_band_height=20):
        """檢測橫向灰條的位置"""
        # 計算每行的平均亮度（但排除垂直亮條區域）
        row_means = []
        for row in range(self.height):
            # 只計算非垂直亮條區域的平均值
            non_vertical_mask = ~self.vertical_lines_mask[row, :]
            if np.any(non_vertical_mask):
                row_mean = np.mean(self.image_np[row, non_vertical_mask])
            else:
                row_mean = np.mean(self.image_np[row, :])
            row_means.append(row_mean)
        
        row_means = np.array(row_means)
        
        # 使用梯度檢測橫條邊界
        gradient = np.gradient(row_means)
        
        # 找出相對穩定的區域（梯度小）
        stable_threshold = np.std(gradient) * 0.5
        stable_regions = np.abs(gradient) < stable_threshold
        
        # 將連續的穩定區域組合成橫條
        self.horizontal_bands = []
        in_band = False
        band_start = 0
        
        for i in range(len(stable_regions)):
            if stable_regions[i] and not in_band:
                in_band = True
                band_start = i
            elif not stable_regions[i] and in_band:
                in_band = False
                if i - band_start >= min_band_height:
                    band_mean = np.mean(row_means[band_start:i])
                    self.horizontal_bands.append({
                        'start': band_start,
                        'end': i,
                        'mean_brightness': band_mean
                    })
        
        # 處理最後一個橫條
        if in_band and len(stable_regions) - band_start >= min_band_height:
            band_mean = np.mean(row_means[band_start:])
            self.horizontal_bands.append({
                'start': band_start,
                'end': len(stable_regions),
                'mean_brightness': band_mean
            })
        
        return self.horizontal_bands
    
    def find_target_band(self, target_bbox):
        """找出目標區域所在的橫條"""
        x, y, w, h = target_bbox
        target_center_y = y + h // 2
        
        # 找出包含目標中心的橫條
        for band in self.horizontal_bands:
            if band['start'] <= target_center_y <= band['end']:
                return band
        
        # 如果沒找到，返回最接近的橫條
        min_dist = float('inf')
        closest_band = None
        for band in self.horizontal_bands:
            band_center = (band['start'] + band['end']) // 2
            dist = abs(band_center - target_center_y)
            if dist < min_dist:
                min_dist = dist
                closest_band = band
        
        return closest_band
    
    def find_reference_regions_in_band(self, target_bbox, target_band, num_refs=5):
        """在目標橫條內尋找參考區域"""
        x, y, w, h = target_bbox
        
        # 在目標橫條內搜尋參考區域
        band_start = target_band['start']
        band_end = target_band['end']
        band_height = band_end - band_start
        
        candidates = []
        
        # 定義參考框大小（可以比目標框大一些，以獲得更穩定的統計）
        ref_width = max(w * 2, 30)  # 至少30像素寬
        ref_height = min(band_height - 10, h * 2)  # 不超過橫條高度
        
        # 在橫條內滑動搜尋
        step_x = ref_width // 2
        
        for col in range(0, self.width - ref_width, step_x):
            # 檢查是否避開垂直亮條
            ref_y = band_start + (band_height - ref_height) // 2
            region_mask = self.vertical_lines_mask[ref_y:ref_y+ref_height, col:col+ref_width]
            vertical_line_ratio = np.sum(region_mask) / (ref_width * ref_height)
            
            # 如果垂直亮條佔比太高，跳過
            if vertical_line_ratio > 0.1:
                continue
            
            # 確保不與目標區域重疊
            if not (col < x + w and col + ref_width > x):
                # 提取候選區域
                candidate_region = self.image_np[ref_y:ref_y+ref_height, col:col+ref_width]
                
                # 計算特徵
                cand_mean = np.mean(candidate_region)
                cand_std = np.std(candidate_region)
                
                candidates.append({
                    'bbox': (col, ref_y, ref_width, ref_height),
                    'mean': cand_mean,
                    'std': cand_std,
                    'distance_from_target': abs(col + ref_width//2 - (x + w//2))
                })
        
        # 按照與目標橫條亮度的相似度排序
        target_band_brightness = target_band['mean_brightness']
        candidates.sort(key=lambda c: abs(c['mean'] - target_band_brightness))
        
        self.reference_regions = candidates[:num_refs]
        return self.reference_regions
    
    def calculate_reference_brightness(self):
        """計算參考區域的平均亮度"""
        if not self.reference_regions:
            return None
        
        brightness_values = []
        for ref in self.reference_regions:
            x, y, w, h = ref['bbox']
            region = self.image_np[y:y+h, x:x+w]
            brightness_values.append(np.mean(region))
        
        # 計算統計值
        ref_mean = np.mean(brightness_values)
        ref_std = np.std(brightness_values)
        ref_median = np.median(brightness_values)
        
        return {
            'mean': ref_mean,
            'std': ref_std,
            'median': ref_median,
            'values': brightness_values
        }
    
    def visualize_results(self, target_bbox, save_path=None):
        """視覺化結果"""
        plt.ioff()  # 關閉互動模式
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 左圖：顯示原圖與所有框
        ax1.imshow(self.image_np, cmap='gray')
        ax1.set_title('Target (Red) and Reference Regions (Green)')
        
        # 繪製橫條區域（淡藍色）
        target_band = self.find_target_band(target_bbox)
        if target_band:
            band_rect = patches.Rectangle((0, target_band['start']), 
                                        self.width, 
                                        target_band['end'] - target_band['start'],
                                        linewidth=1, edgecolor='cyan', 
                                        facecolor='cyan', alpha=0.1)
            ax1.add_patch(band_rect)
        
        # 繪製目標框（紅色）
        x, y, w, h = target_bbox
        target_rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                      edgecolor='red', facecolor='none')
        ax1.add_patch(target_rect)
        ax1.text(x+5, y-5, f'Target\nPos:({x},{y})\nSize:{w}x{h}', 
                color='red', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="white", alpha=0.8))
        
        # 繪製參考框（綠色）
        for i, ref in enumerate(self.reference_regions):
            x, y, w, h = ref['bbox']
            ref_rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                       edgecolor='green', facecolor='none')
            ax1.add_patch(ref_rect)
            ax1.text(x+5, y-5, f'Ref{i+1}\nBright:{ref["mean"]:.1f}', 
                    color='green', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="white", alpha=0.8))
        
        # 右圖：顯示垂直線條和橫條檢測結果
        ax2.imshow(self.vertical_lines_mask, cmap='gray')
        ax2.set_title('Vertical Lines (White) and Horizontal Bands')
        
        # 在右圖上繪製橫條邊界
        for band in self.horizontal_bands:
            ax2.axhline(y=band['start'], color='yellow', linewidth=1, alpha=0.7)
            ax2.axhline(y=band['end'], color='yellow', linewidth=1, alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()  # 關閉圖形，不顯示
        
        # 打印詳細信息
        print("\n=== Analysis Results ===")
        print(f"Target Region: Position({target_bbox[0]}, {target_bbox[1]}), Size {target_bbox[2]}x{target_bbox[3]}")
        
        if target_band:
            print(f"Target Band: Row {target_band['start']}-{target_band['end']}, "
                  f"Mean Brightness={target_band['mean_brightness']:.1f}")
        
        print(f"\nFound {len(self.reference_regions)} reference regions:")
        for i, ref in enumerate(self.reference_regions):
            print(f"  Reference {i+1}: Position({ref['bbox'][0]}, {ref['bbox'][1]}), "
                  f"Size {ref['bbox'][2]}x{ref['bbox'][3]}, "
                  f"Mean Brightness={ref['mean']:.1f}")
        
        # 計算參考亮度
        ref_brightness = self.calculate_reference_brightness()
        if ref_brightness:
            print(f"\nReference Brightness Statistics:")
            print(f"  Mean: {ref_brightness['mean']:.2f}")
            print(f"  Std Dev: {ref_brightness['std']:.2f}")
            print(f"  Median: {ref_brightness['median']:.2f}")
            
        # 計算目標區域亮度
        x, y, w, h = target_bbox
        target_region = self.image_np[y:y+h, x:x+w]
        target_brightness = np.mean(target_region)
        print(f"\nTarget Region Brightness: {target_brightness:.2f}")
        
        if ref_brightness:
            brightness_diff = target_brightness - ref_brightness['mean']
            brightness_ratio = target_brightness / ref_brightness['mean']
            print(f"Brightness Difference: {brightness_diff:.2f} (vs reference mean)")
            print(f"Brightness Ratio: {brightness_ratio:.2f}x (target/reference)")


# 測試程式碼
if __name__ == "__main__":
    # 初始化分析器
    analyzer = DefectAnalyzer('image_with_bump.jpg')
    
    # 檢測垂直亮條
    analyzer.detect_vertical_lines(threshold_factor=1.2)
    
    # 檢測橫條
    analyzer.detect_horizontal_bands(min_band_height=20)
    
    # 使用者框選的缺陷區域
    # 缺陷左上角 (548, 494)，右下角 (560, 527)
    x1, y1 = 548, 494
    x2, y2 = 560, 527
    target_bbox = (x1, y1, x2-x1, y2-y1)  # (x, y, width, height) = (548, 494, 12, 33)
    
    # 找出目標所在的橫條
    target_band = analyzer.find_target_band(target_bbox)
    
    # 在同一橫條內尋找參考區域
    analyzer.find_reference_regions_in_band(target_bbox, target_band, num_refs=5)
    
    # 視覺化結果
    analyzer.visualize_results(target_bbox, save_path='defect_analysis_result_v2.png')