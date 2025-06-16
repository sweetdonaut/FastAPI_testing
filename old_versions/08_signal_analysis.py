import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from reference_finder import ReferenceFinder

class SignalAnalyzer:
    def __init__(self, image_path):
        """初始化訊號分析器"""
        self.finder = ReferenceFinder(image_path)
        self.finder.detect_grid_lines()
        self.finder.create_grid_cells()
        self.finder.classify_cells()
        
    def analyze_signal(self, target_bbox):
        """分析目標訊號相對於參考的強度"""
        # 找出目標單元
        target_cell = self.finder.find_target_cell(target_bbox)
        
        # 找出參考單元
        self.finder.find_reference_cells(target_cell, edge_margin=100, 
                                        brightness_tolerance=10, size_tolerance=20)
        
        # 計算目標區域的精確亮度（使用原始bbox，不是整個cell）
        x, y, w, h = target_bbox
        target_region = self.finder.image_np[y:y+h, x:x+w]
        target_mean = np.mean(target_region)
        target_std = np.std(target_region)
        target_median = np.median(target_region)
        
        # 計算所有參考區域的亮度
        reference_values = []
        for ref_cell in self.finder.reference_cells:
            # 在參考cell中選取與目標框相同大小的區域
            cell_center_x = (ref_cell['x1'] + ref_cell['x2']) // 2
            cell_center_y = (ref_cell['y1'] + ref_cell['y2']) // 2
            
            # 確保不超出cell邊界
            ref_x = max(ref_cell['x1'], min(cell_center_x - w//2, ref_cell['x2'] - w))
            ref_y = max(ref_cell['y1'], min(cell_center_y - h//2, ref_cell['y2'] - h))
            
            ref_region = self.finder.image_np[ref_y:ref_y+h, ref_x:ref_x+w]
            reference_values.append(np.mean(ref_region))
        
        # 計算參考統計
        ref_mean = np.mean(reference_values)
        ref_std = np.std(reference_values)
        ref_median = np.median(reference_values)
        
        # 計算訊號強度指標
        signal_ratio = target_mean / ref_mean
        signal_diff = target_mean - ref_mean
        snr = signal_diff / ref_std if ref_std > 0 else float('inf')  # Signal-to-Noise Ratio
        
        # Z-score: 目標偏離參考多少個標準差
        z_score = (target_mean - ref_mean) / ref_std if ref_std > 0 else float('inf')
        
        results = {
            'target': {
                'bbox': target_bbox,
                'mean': target_mean,
                'std': target_std,
                'median': target_median
            },
            'reference': {
                'mean': ref_mean,
                'std': ref_std,
                'median': ref_median,
                'num_refs': len(reference_values),
                'values': reference_values
            },
            'signal': {
                'ratio': signal_ratio,
                'difference': signal_diff,
                'snr': snr,
                'z_score': z_score
            }
        }
        
        return results
    
    def visualize_signal_analysis(self, target_bbox, results, save_path=None):
        """視覺化訊號分析結果"""
        plt.ioff()
        fig = plt.figure(figsize=(16, 10))
        
        # 創建子圖布局
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[2, 1])
        ax_main = fig.add_subplot(gs[0, :])
        ax_hist = fig.add_subplot(gs[1, 0])
        ax_box = fig.add_subplot(gs[1, 1])
        ax_info = fig.add_subplot(gs[2, :])
        
        # 主圖：顯示目標和參考區域
        ax_main.imshow(self.finder.image_np, cmap='gray')
        ax_main.set_title('Signal Analysis: Target (Red) vs References (Green)', fontsize=14)
        
        # 繪製目標框
        x, y, w, h = target_bbox
        target_rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                      edgecolor='red', facecolor='none')
        ax_main.add_patch(target_rect)
        ax_main.text(x+w+5, y+h/2, f"Target\nBrightness: {results['target']['mean']:.1f}", 
                    color='red', fontsize=10, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 繪製參考框（只顯示前10個）
        for i, ref_cell in enumerate(self.finder.reference_cells[:10]):
            rect = patches.Rectangle((ref_cell['x1'], ref_cell['y1']), 
                                   ref_cell['width'], ref_cell['height'],
                                   linewidth=1.5, edgecolor='green', 
                                   facecolor='green', alpha=0.1)
            ax_main.add_patch(rect)
        
        # 直方圖：亮度分布
        ax_hist.hist(results['reference']['values'], bins=20, alpha=0.7, 
                    color='green', label='Reference', density=True)
        ax_hist.axvline(results['reference']['mean'], color='green', 
                       linestyle='--', linewidth=2, label=f'Ref mean: {results["reference"]["mean"]:.1f}')
        ax_hist.axvline(results['target']['mean'], color='red', 
                       linestyle='-', linewidth=3, label=f'Target: {results["target"]["mean"]:.1f}')
        ax_hist.set_xlabel('Brightness')
        ax_hist.set_ylabel('Density')
        ax_hist.set_title('Brightness Distribution')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # 箱形圖：比較目標和參考
        box_data = [results['reference']['values'], [results['target']['mean']]]
        bp = ax_box.boxplot(box_data, labels=['Reference', 'Target'], 
                           patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor('red')
        bp['boxes'][1].set_alpha(0.5)
        ax_box.set_ylabel('Brightness')
        ax_box.set_title('Statistical Comparison')
        ax_box.grid(True, alpha=0.3)
        
        # 信息面板
        ax_info.axis('off')
        
        info_text = "=== SIGNAL ANALYSIS RESULTS ===\n\n"
        
        info_text += "Target Region:\n"
        info_text += f"  Position: ({target_bbox[0]}, {target_bbox[1]})\n"
        info_text += f"  Size: {target_bbox[2]}x{target_bbox[3]}\n"
        info_text += f"  Brightness: {results['target']['mean']:.2f} ± {results['target']['std']:.2f}\n\n"
        
        info_text += f"Reference Statistics ({results['reference']['num_refs']} regions):\n"
        info_text += f"  Mean brightness: {results['reference']['mean']:.2f} ± {results['reference']['std']:.2f}\n"
        info_text += f"  Range: {min(results['reference']['values']):.1f} - {max(results['reference']['values']):.1f}\n\n"
        
        info_text += "Signal Metrics:\n"
        info_text += f"  Brightness Ratio: {results['signal']['ratio']:.3f}x\n"
        info_text += f"  Absolute Difference: +{results['signal']['difference']:.2f}\n"
        info_text += f"  Signal-to-Noise Ratio: {results['signal']['snr']:.2f}\n"
        info_text += f"  Z-Score: {results['signal']['z_score']:.2f}σ\n\n"
        
        # 判斷缺陷類型
        if results['signal']['z_score'] > 3:
            info_text += ">> DEFECT DETECTED: Bright spot (>3σ from reference)\n"
            defect_type = "BRIGHT DEFECT"
            defect_color = "red"
        elif results['signal']['z_score'] < -3:
            info_text += ">> DEFECT DETECTED: Dark spot (<-3σ from reference)\n"
            defect_type = "DARK DEFECT"
            defect_color = "blue"
        else:
            info_text += ">> NORMAL: Within acceptable range (±3σ)\n"
            defect_type = "NORMAL"
            defect_color = "green"
        
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 添加結論標籤
        ax_info.text(0.7, 0.5, defect_type, transform=ax_info.transAxes,
                    fontsize=24, weight='bold', color=defect_color,
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                            edgecolor=defect_color, linewidth=3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 打印結果
        print(info_text)


# 主程式
if __name__ == "__main__":
    # 初始化分析器
    analyzer = SignalAnalyzer('image_with_bump.jpg')
    
    # 使用者框選的缺陷區域
    x1, y1 = 548, 494
    x2, y2 = 560, 527
    target_bbox = (x1, y1, x2-x1, y2-y1)
    
    # 執行訊號分析
    results = analyzer.analyze_signal(target_bbox)
    
    # 視覺化結果
    analyzer.visualize_signal_analysis(target_bbox, results, 
                                     save_path='signal_analysis_result.png')