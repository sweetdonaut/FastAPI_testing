import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class GlobalDefectAnalyzer:
    def __init__(self, image_path):
        """初始化分析器"""
        self.image = Image.open(image_path).convert('L')
        self.image_np = np.array(self.image)
        self.height, self.width = self.image_np.shape
        self.vertical_lines_mask = None
        self.horizontal_bands = []
        self.reference_regions = []
        self.similar_bands = []
        
    def detect_vertical_lines(self, threshold_factor=1.2):
        """檢測垂直亮條的位置"""
        column_means = np.mean(self.image_np, axis=0)
        mean_brightness = np.mean(column_means)
        threshold = mean_brightness * threshold_factor
        
        self.vertical_lines_mask = np.zeros_like(self.image_np, dtype=bool)
        for col in range(self.width):
            if column_means[col] > threshold:
                self.vertical_lines_mask[:, col] = True
                
        return self.vertical_lines_mask
    
    def detect_horizontal_bands(self, min_band_height=20):
        """檢測橫向灰條的位置"""
        row_means = []
        for row in range(self.height):
            non_vertical_mask = ~self.vertical_lines_mask[row, :]
            if np.any(non_vertical_mask):
                row_mean = np.mean(self.image_np[row, non_vertical_mask])
            else:
                row_mean = np.mean(self.image_np[row, :])
            row_means.append(row_mean)
        
        row_means = np.array(row_means)
        gradient = np.gradient(row_means)
        stable_threshold = np.std(gradient) * 0.5
        stable_regions = np.abs(gradient) < stable_threshold
        
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
                        'mean_brightness': band_mean,
                        'height': i - band_start
                    })
        
        if in_band and len(stable_regions) - band_start >= min_band_height:
            band_mean = np.mean(row_means[band_start:])
            self.horizontal_bands.append({
                'start': band_start,
                'end': len(stable_regions),
                'mean_brightness': band_mean,
                'height': len(stable_regions) - band_start
            })
        
        return self.horizontal_bands
    
    def find_target_band(self, target_bbox):
        """找出目標區域所在的橫條"""
        x, y, w, h = target_bbox
        target_center_y = y + h // 2
        
        for band in self.horizontal_bands:
            if band['start'] <= target_center_y <= band['end']:
                return band
        
        min_dist = float('inf')
        closest_band = None
        for band in self.horizontal_bands:
            band_center = (band['start'] + band['end']) // 2
            dist = abs(band_center - target_center_y)
            if dist < min_dist:
                min_dist = dist
                closest_band = band
        
        return closest_band
    
    def find_similar_bands(self, target_band, brightness_tolerance=5, height_tolerance=10):
        """找出全圖中與目標橫條相似的所有橫條"""
        self.similar_bands = []
        
        target_brightness = target_band['mean_brightness']
        target_height = target_band['height']
        
        for band in self.horizontal_bands:
            # 檢查亮度和高度是否相似
            brightness_diff = abs(band['mean_brightness'] - target_brightness)
            height_diff = abs(band['height'] - target_height)
            
            if brightness_diff <= brightness_tolerance and height_diff <= height_tolerance:
                self.similar_bands.append(band)
        
        return self.similar_bands
    
    def find_reference_regions_global(self, target_bbox, refs_per_band=3, max_total_refs=20):
        """在所有相似橫條中尋找參考區域"""
        x, y, w, h = target_bbox
        self.reference_regions = []
        
        # 定義參考框大小
        ref_width = max(w * 2, 30)
        
        for band in self.similar_bands:
            band_height = band['end'] - band['start']
            ref_height = min(band_height - 10, h * 2)
            
            if ref_height <= 0:
                continue
            
            # 在每個橫條中找參考區域
            band_refs = []
            step_x = self.width // (refs_per_band + 1)  # 均勻分布
            
            for i in range(1, refs_per_band + 1):
                col = i * step_x - ref_width // 2
                
                # 確保在圖像範圍內
                col = max(0, min(col, self.width - ref_width))
                
                # 檢查是否避開垂直亮條
                ref_y = band['start'] + (band_height - ref_height) // 2
                region_mask = self.vertical_lines_mask[ref_y:ref_y+ref_height, col:col+ref_width]
                vertical_line_ratio = np.sum(region_mask) / (ref_width * ref_height)
                
                if vertical_line_ratio > 0.1:
                    # 嘗試左右移動找到沒有垂直線的位置
                    for offset in [-50, 50, -100, 100]:
                        new_col = col + offset
                        if 0 <= new_col <= self.width - ref_width:
                            region_mask = self.vertical_lines_mask[ref_y:ref_y+ref_height, new_col:new_col+ref_width]
                            vertical_line_ratio = np.sum(region_mask) / (ref_width * ref_height)
                            if vertical_line_ratio <= 0.1:
                                col = new_col
                                break
                
                # 確保不與目標區域重疊（如果在同一橫條）
                if band == self.find_target_band(target_bbox):
                    if col < x + w and col + ref_width > x:
                        continue
                
                # 提取候選區域
                candidate_region = self.image_np[ref_y:ref_y+ref_height, col:col+ref_width]
                
                band_refs.append({
                    'bbox': (col, ref_y, ref_width, ref_height),
                    'mean': np.mean(candidate_region),
                    'std': np.std(candidate_region),
                    'band': band,
                    'band_index': self.similar_bands.index(band)
                })
            
            self.reference_regions.extend(band_refs)
            
            if len(self.reference_regions) >= max_total_refs:
                self.reference_regions = self.reference_regions[:max_total_refs]
                break
        
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
        
        return {
            'mean': np.mean(brightness_values),
            'std': np.std(brightness_values),
            'median': np.median(brightness_values),
            'values': brightness_values
        }
    
    def visualize_results(self, target_bbox, save_path=None):
        """視覺化結果"""
        plt.ioff()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左圖：顯示原圖與所有框
        ax1.imshow(self.image_np, cmap='gray')
        ax1.set_title('Global Reference Regions Analysis')
        
        # 繪製所有相似橫條（淡藍色）
        for i, band in enumerate(self.similar_bands):
            band_rect = patches.Rectangle((0, band['start']), 
                                        self.width, 
                                        band['end'] - band['start'],
                                        linewidth=0.5, edgecolor='cyan', 
                                        facecolor='cyan', alpha=0.05)
            ax1.add_patch(band_rect)
            # 標記橫條編號
            ax1.text(5, (band['start'] + band['end']) / 2, f'B{i+1}', 
                    color='cyan', fontsize=6, weight='bold')
        
        # 繪製目標框（紅色）
        x, y, w, h = target_bbox
        target_rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                      edgecolor='red', facecolor='none')
        ax1.add_patch(target_rect)
        ax1.text(x+w+5, y+h/2, f'Target\n({x},{y})', 
                color='red', fontsize=8, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 繪製參考框（綠色，按橫條分組）
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.similar_bands)))
        for i, ref in enumerate(self.reference_regions):
            band_idx = ref['band_index']
            x, y, w, h = ref['bbox']
            ref_rect = patches.Rectangle((x, y), w, h, linewidth=1.5, 
                                       edgecolor=colors[band_idx], facecolor='none')
            ax1.add_patch(ref_rect)
            # 簡化標籤
            ax1.text(x+2, y+2, f'R{i+1}', 
                    color=colors[band_idx], fontsize=6, weight='bold')
        
        # 右圖：顯示統計信息
        ax2.axis('off')
        
        # 顯示分析結果
        result_text = "=== ANALYSIS RESULTS ===\n\n"
        
        # 目標信息
        target_band = self.find_target_band(target_bbox)
        result_text += f"Target Region:\n"
        result_text += f"  Position: ({target_bbox[0]}, {target_bbox[1]})\n"
        result_text += f"  Size: {target_bbox[2]}x{target_bbox[3]}\n"
        result_text += f"  Band: Row {target_band['start']}-{target_band['end']}\n\n"
        
        # 相似橫條信息
        result_text += f"Found {len(self.similar_bands)} similar bands:\n"
        for i, band in enumerate(self.similar_bands[:5]):  # 只顯示前5個
            result_text += f"  Band {i+1}: Row {band['start']}-{band['end']}, "
            result_text += f"Brightness={band['mean_brightness']:.1f}\n"
        if len(self.similar_bands) > 5:
            result_text += f"  ... and {len(self.similar_bands)-5} more\n"
        
        # 參考區域統計
        result_text += f"\nTotal Reference Regions: {len(self.reference_regions)}\n"
        
        # 亮度統計
        ref_brightness = self.calculate_reference_brightness()
        if ref_brightness:
            result_text += f"\nReference Brightness:\n"
            result_text += f"  Mean: {ref_brightness['mean']:.2f}\n"
            result_text += f"  Std Dev: {ref_brightness['std']:.2f}\n"
            result_text += f"  Range: {min(ref_brightness['values']):.2f} - {max(ref_brightness['values']):.2f}\n"
        
        # 目標亮度
        tx, ty, tw, th = target_bbox
        target_region = self.image_np[ty:ty+th, tx:tx+tw]
        target_brightness = np.mean(target_region)
        result_text += f"\nTarget Brightness: {target_brightness:.2f}\n"
        
        if ref_brightness:
            brightness_ratio = target_brightness / ref_brightness['mean']
            result_text += f"Brightness Ratio: {brightness_ratio:.2f}x\n"
            
            # 判斷是否為缺陷
            if brightness_ratio > 1.3:
                result_text += f"\n>> DEFECT DETECTED (Bright spot)"
            elif brightness_ratio < 0.7:
                result_text += f"\n>> DEFECT DETECTED (Dark spot)"
            else:
                result_text += f"\n>> NORMAL REGION"
        
        ax2.text(0.05, 0.95, result_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 同時輸出到終端
        print(result_text)


# 測試程式碼
if __name__ == "__main__":
    # 初始化分析器
    analyzer = GlobalDefectAnalyzer('image_with_bump.jpg')
    
    # 檢測垂直亮條
    analyzer.detect_vertical_lines(threshold_factor=1.2)
    
    # 檢測橫條
    analyzer.detect_horizontal_bands(min_band_height=20)
    
    # 使用者框選的缺陷區域
    x1, y1 = 548, 494
    x2, y2 = 560, 527
    target_bbox = (x1, y1, x2-x1, y2-y1)
    
    # 找出目標所在的橫條
    target_band = analyzer.find_target_band(target_bbox)
    
    # 找出全圖中相似的橫條
    analyzer.find_similar_bands(target_band, brightness_tolerance=5, height_tolerance=15)
    
    # 在所有相似橫條中尋找參考區域
    analyzer.find_reference_regions_global(target_bbox, refs_per_band=3, max_total_refs=20)
    
    # 視覺化結果
    analyzer.visualize_results(target_bbox, save_path='defect_analysis_global_result.png')