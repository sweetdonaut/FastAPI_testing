import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DefectAnalyzer:
    def __init__(self, image_path):
        """初始化分析器"""
        self.image = Image.open(image_path).convert('L')
        self.image_np = np.array(self.image)
        self.height, self.width = self.image_np.shape
        self.vertical_lines_mask = None
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
    
    def find_reference_regions(self, bbox, num_refs=5, similarity_threshold=0.7):
        """
        尋找參考區域
        bbox: (x, y, width, height) - 目標區域的邊界框
        """
        x, y, w, h = bbox
        target_region = self.image_np[y:y+h, x:x+w]
        
        # 計算目標區域的特徵
        target_mean = np.mean(target_region)
        target_std = np.std(target_region)
        
        # 候選區域列表
        candidates = []
        
        # 滑動窗口搜尋
        step_size = min(w//2, h//2)  # 步長為窗口大小的一半
        
        for row in range(0, self.height - h, step_size):
            for col in range(0, self.width - w, step_size):
                # 檢查是否避開垂直亮條
                region_mask = self.vertical_lines_mask[row:row+h, col:col+w]
                vertical_line_ratio = np.sum(region_mask) / (w * h)
                
                # 如果垂直亮條佔比太高，跳過此區域
                if vertical_line_ratio > 0.3:
                    continue
                
                # 提取候選區域
                candidate_region = self.image_np[row:row+h, col:col+w]
                
                # 計算候選區域特徵
                cand_mean = np.mean(candidate_region)
                cand_std = np.std(candidate_region)
                
                # 計算相似度（基於統計特徵）
                mean_diff = abs(cand_mean - target_mean) / target_mean
                std_diff = abs(cand_std - target_std) / (target_std + 1e-6)
                
                # 綜合相似度評分
                similarity = 1 - (mean_diff + std_diff) / 2
                
                # 如果相似度足夠高且不是目標區域本身
                if similarity > similarity_threshold:
                    # 確保不與目標區域重疊
                    if not (col < x + w and col + w > x and row < y + h and row + h > y):
                        candidates.append({
                            'bbox': (col, row, w, h),
                            'similarity': similarity,
                            'mean': cand_mean,
                            'std': cand_std
                        })
        
        # 按相似度排序，選擇最佳的參考區域
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
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
        ax1.set_title('目標區域（紅框）與參考區域（綠框）')
        
        # 繪製目標框（紅色）
        x, y, w, h = target_bbox
        target_rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                      edgecolor='red', facecolor='none')
        ax1.add_patch(target_rect)
        ax1.text(x+5, y-5, f'目標區域\n位置:({x},{y})\n大小:{w}x{h}', 
                color='red', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="white", alpha=0.8))
        
        # 繪製參考框（綠色）
        for i, ref in enumerate(self.reference_regions):
            x, y, w, h = ref['bbox']
            ref_rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                       edgecolor='green', facecolor='none')
            ax1.add_patch(ref_rect)
            ax1.text(x+5, y-5, f'參考{i+1}\n亮度:{ref["mean"]:.1f}', 
                    color='green', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="white", alpha=0.8))
        
        # 右圖：顯示垂直線條遮罩
        ax2.imshow(self.vertical_lines_mask, cmap='gray')
        ax2.set_title('垂直亮條檢測結果（白色為亮條）')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()  # 關閉圖形，不顯示
        
        # 打印詳細信息
        print("\n=== 分析結果 ===")
        print(f"目標區域：位置({target_bbox[0]}, {target_bbox[1]})，大小 {target_bbox[2]}x{target_bbox[3]}")
        print(f"找到 {len(self.reference_regions)} 個參考區域：")
        for i, ref in enumerate(self.reference_regions):
            print(f"  參考區域{i+1}: 位置({ref['bbox'][0]}, {ref['bbox'][1]})，"
                  f"相似度={ref['similarity']:.3f}，平均亮度={ref['mean']:.1f}")
        
        # 計算參考亮度
        ref_brightness = self.calculate_reference_brightness()
        if ref_brightness:
            print(f"\n參考亮度統計：")
            print(f"  平均值: {ref_brightness['mean']:.2f}")
            print(f"  標準差: {ref_brightness['std']:.2f}")
            print(f"  中位數: {ref_brightness['median']:.2f}")
            
        # 計算目標區域亮度
        x, y, w, h = target_bbox
        target_region = self.image_np[y:y+h, x:x+w]
        target_brightness = np.mean(target_region)
        print(f"\n目標區域亮度: {target_brightness:.2f}")
        
        if ref_brightness:
            brightness_diff = target_brightness - ref_brightness['mean']
            brightness_ratio = target_brightness / ref_brightness['mean']
            print(f"亮度差異: {brightness_diff:.2f} (相對於參考平均值)")
            print(f"亮度比例: {brightness_ratio:.2f}x (目標/參考)")


# 測試程式碼
if __name__ == "__main__":
    # 初始化分析器
    analyzer = DefectAnalyzer('image_with_bump.jpg')
    
    # 檢測垂直亮條
    analyzer.detect_vertical_lines(threshold_factor=1.2)
    
    # 使用者框選的缺陷區域
    # 缺陷左上角 (548, 494)，右下角 (560, 527)
    x1, y1 = 548, 494
    x2, y2 = 560, 527
    target_bbox = (x1, y1, x2-x1, y2-y1)  # (x, y, width, height) = (548, 494, 12, 33)
    
    # 尋找參考區域
    analyzer.find_reference_regions(target_bbox, num_refs=5, similarity_threshold=0.6)
    
    # 視覺化結果
    analyzer.visualize_results(target_bbox, save_path='defect_analysis_result.png')