# SEM 缺陷檢測系統

## 系統概述
本系統用於自動檢測 SEM（掃描式電子顯微鏡）影像中的缺陷，通過智能選擇參考區域並進行統計分析來判斷異常。

## 第一階段完成
經過多次迭代開發，我們成功完成了第一階段的缺陷檢測系統，具備完整的分析和視覺化功能。

## 主要程式檔案

### 1. sem_defect_detector_v1.py - v1 版本（推薦使用）
**功能**：
- 完整的缺陷檢測分析流程
- 清晰的視覺化展示
- 右側面板顯示所有關鍵數據

**使用方式**：
```python
python sem_defect_detector_v1.py
```

**輸出內容**：
- 目標框位置和亮度
- 參考區域統計（數量、平均亮度、範圍）
- 亮度比例和缺陷判斷
- 視覺化圖片：sem_defect_analysis_v1.png

### 2. 舊版本檔案（已移至 old_versions/）
原始的 `reference_finder.py` 和 `reference_analysis_final.py` 已被移至 old_versions/ 資料夾：
- `10_reference_analysis_final.py` - 原最終整合版本
- `10_reference_finder.py` - 原核心模組
- `10_reference_analysis_final.png` - 原分析結果

### 核心模組功能說明（參考 v1 版本實作）
**功能**：
- 自動檢測圖片網格結構
- 識別灰色方塊、白色垂直條、暗區
- 智能選擇合適的參考區域
- 排除邊界干擾區域

**使用方式**：
```python
# 初始化分析器
finder = ReferenceFinder('image.jpg')

# 檢測網格
finder.detect_grid_lines()
finder.create_grid_cells()
finder.classify_cells()

# 定義目標區域
target_bbox = (x1, y1, width, height)

# 找出參考區域
target_cell = finder.find_target_cell(target_bbox)
finder.find_reference_cells(target_cell, 
                           edge_margin=100,      # 邊界安全距離
                           brightness_tolerance=10,  # 亮度容差
                           size_tolerance=20)       # 尺寸容差

# 視覺化結果
finder.visualize_references(target_bbox, save_path='result.png')
```


## 參數說明

### 參考區域選擇參數
- **edge_margin**: 邊界安全距離（預設100px）
- **brightness_tolerance**: 亮度容差（預設±10）
- **size_tolerance**: 尺寸容差（預設±20px）

### 缺陷判斷標準
- **正常**: Z分數在±3σ範圍內
- **亮點缺陷**: Z分數 > 3σ
- **暗點缺陷**: Z分數 < -3σ

## 分析結果說明

### v1 版本輸出（sem_defect_analysis_v1.png）
1. **視覺化元素**：
   - 紅色粗框：目標缺陷位置
   - 綠色區域：被選中的參考區域
   - 黃色區域：因太靠近邊界被排除
   - 橙色區域：因亮度差異太大被排除

2. **數據面板內容**：
   - 目標框資訊（位置、大小、亮度）
   - 參考分析（數量、平均亮度、範圍）
   - 訊號分析（亮度比例、差異值）
   - 缺陷判斷結果

3. **缺陷判斷標準**：
   - 亮度比例 > 1.3：亮點缺陷
   - 亮度比例 < 0.7：暗點缺陷
   - 其他：正常

## 實際案例結果
- 目標亮度：79.63
- 參考平均：50.39（46個區域）
- 亮度比例：1.580×
- 判斷：**亮點缺陷**

## 歷史版本
詳細的開發歷程請參考 `discussion_history.md`
舊版本程式碼保存在 `old_versions/` 資料夾中（共11個迭代版本）