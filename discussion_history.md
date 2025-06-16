# SEM 缺陷檢測演算法開發討論歷程

## 專案概述
開發一個自動檢測 SEM（掃描式電子顯微鏡）影像中缺陷的演算法，透過比較目標區域與參考區域的亮度差異來判斷是否存在缺陷。

## 第一階段完成
我們成功開發了一個完整的缺陷檢測系統，能夠：
1. 自動識別圖片網格結構
2. 智能選擇參考區域（排除邊界和異常值）
3. 計算目標與參考的亮度比例
4. 自動判斷缺陷類型

## 討論時間軸與版本演進

### 第一階段：初步分析與垂直條檢測
**主要目標**：理解圖片結構，找出直條和橫條的特徵

#### 1. 缺陷分析 v1（defect_analysis.py）
- **功能**：檢測垂直亮條，避開亮條選擇參考區域
- **問題**：目標框位置錯誤，參考區域選擇不夠準確
- **輸出**：defect_analysis_result.png

### 第二階段：改進參考區域選擇
**主要目標**：基於橫條位置選擇參考區域

#### 2. 缺陷分析 v2（defect_analysis_v2.py）
- **功能**：
  - 檢測橫條結構
  - 在同一橫條內選擇參考區域
  - 參考框大小可調整
- **改進**：修正目標框位置到 (548, 494) - (560, 527)
- **輸出**：defect_analysis_result_v2.png

#### 3. 全圖缺陷分析（defect_analysis_global.py）
- **功能**：搜尋全圖所有相似橫條作為參考
- **問題**：沒有正確識別灰色方塊結構
- **輸出**：defect_analysis_global_result.png

### 第三階段：網格結構分析（重要突破）
**主要目標**：準確找出目標框所在的灰色方塊

#### 4. 網格分析（grid_analysis.py）
- **功能**：
  - 檢測網格線（垂直和水平）
  - 識別所有網格單元
  - 分類單元類型（亮條、灰色方塊、暗區）
- **成果**：成功找到目標在第8行第8列的灰色方塊
- **輸出**：grid_analysis_result.png

### 第四階段：參考區域智能選擇
**主要目標**：排除邊界干擾，選擇合適的參考區域

#### 5. 參考區域搜尋器（reference_finder.py）
- **功能**：
  - 根據目標灰色方塊特徵搜尋相似方塊
  - 排除太靠近邊界的方塊（>100px）
  - 亮度和尺寸相似性篩選
- **參數**：
  - 邊界距離：>100px
  - 亮度容差：±10
  - 尺寸容差：±20px
- **成果**：找到46個合適的參考區域
- **輸出**：reference_selection_result.png

#### 6. 詳細分析視覺化（reference_analysis_detailed.py）
- **功能**：視覺化顯示為什麼右半邊很多框沒被選上
- **發現**：
  - 黃色框：太靠近邊界
  - 橙色框：亮度差異太大（白色垂直條）
- **輸出**：reference_analysis_detailed.png

### 第五階段：訊號分析
**主要目標**：計算目標框相對於參考框的訊號強度

#### 7. 參考區域選擇器最終版（reference_finder.py）
- **功能**：智能選擇參考區域的核心模組
- **輸出**：reference_selection_result.png

#### 8. 訊號分析器（signal_analysis.py）
- **功能**：
  - 使用已選定的參考區域計算基準
  - 計算多種訊號指標（比例、SNR、Z分數）
  - 自動判斷缺陷類型
- **結果**：
  - 目標亮度：79.63 ± 22.51
  - 參考亮度：50.39 ± 1.67
  - Z分數：17.47σ（確定為亮點缺陷）
- **輸出**：signal_analysis_result.png

#### 9. 詳細分析視覺化複製版（reference_analysis_detailed.py）
- **功能**：從第5版複製出來，用於詳細視覺化
- **輸出**：reference_analysis_detailed.png

### 第六階段：最終整合版本（第一階段完成）
**主要目標**：整合所有功能，提供完整的分析視圖

#### 10. 最終分析版本（reference_analysis_final.py）- 保留在外
- **功能**：
  - 顯示目標物件框（紅色粗框）
  - 右側面板顯示完整分析結果
  - 包含目標亮度、參考統計、亮度比例
- **結果**：
  - 目標框亮度：79.63
  - 46個參考區域平均亮度：50.39
  - 亮度比例：1.580×
  - 判斷：亮點缺陷
- **輸出**：reference_analysis_final.png

## 關鍵技術突破

1. **網格結構識別**：準確檢測圖片的網格結構，識別灰色方塊
2. **智能參考選擇**：排除邊界和異常區域，確保參考值可靠性
3. **統計分析方法**：使用Z分數判斷缺陷，提供客觀的檢測標準

## 最終成果

- **檢測準確度**：成功檢測到亮點缺陷（偏離17.47個標準差）
- **參考區域**：46個經過篩選的灰色方塊，確保統計可靠性
- **視覺化**：清晰展示分析過程和結果

## 檔案結構
```
FastAPI_testing/
├── sem_defect_detector_v1.py         # v1 版本：完整分析程式
├── sem_defect_analysis_v1.png        # v1 版本：分析結果圖
├── discussion_history.md             # 本文件
├── README_defect_detection.md        # 使用說明
└── old_versions/                     # 歷史版本（共11個迭代）
    ├── 01_defect_analysis_v1.py
    ├── 01_defect_analysis_result.png
    ├── 02_defect_analysis_v2.py
    ├── 02_defect_analysis_result_v2.png
    ├── 03_defect_analysis_global.py
    ├── 03_defect_analysis_global_result.png
    ├── 04_grid_analysis.py
    ├── 04_grid_analysis_result.png
    ├── 05_reference_analysis_detailed.py
    ├── 05_reference_analysis_detailed.png
    ├── 06_reference_selection_result.png
    ├── 07_reference_finder.py
    ├── 07_reference_selection_result.png
    ├── 08_signal_analysis.py
    ├── 08_signal_analysis_result.png
    ├── 09_reference_analysis_detailed.py
    ├── 09_reference_analysis_detailed.png
    ├── 10_reference_finder.py              # 原核心模組
    ├── 10_reference_analysis_final.py      # 原最終版
    └── 10_reference_analysis_final.png     # 原最終版結果圖
```