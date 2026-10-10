# README 圖片 SVG 重製紀錄

已檢查 12 章的 42 張主要圖片。23 張流程圖、比較表與矩陣圖以真正的 SVG 圖形和文字重製；19 張包含散點、曲線或複雜座標的圖片保留 PNG。所有原始 PNG 都保留，README 在 SVG 下提供原圖連結。

這是依原圖與教材內容重新排版的向量版，並非逐像素轉換。保留核心教學關係與原圖數據；較長的說明整理成短句，完整教材仍在各章 README，原 PNG 可隨時對照。

主要圖片檔案合計由 13,690,036 bytes 降到 6,103,334 bytes，減少 55.4%。不包含 README、網頁程式與字型；未以線上網速實測載入秒數。

## 維護方式

```sh
.venv/bin/python 名詞解釋/tools/build_svg_diagrams.py
.venv/bin/python docs/glossary/build.py
```

向量圖來源：`tools/build_svg_diagrams.py`。PNG 與 SVG 的對照：`tools/svg_map.json`。各圖位於原章節 `images/` 中，檔名保持相同，只改副檔名。網頁場景的 `source_images` 同步改為 SVG，避免失去動畫與原圖的對應。

SVG 使用本機中文字型，不內嵌字型或點陣圖片，也沒有 JavaScript、外部圖片或 foreignObject。不同系統字型可能略有差異；文字可縮放，無須下載額外字型。

## 每張圖片的處理

| 章節 | 原圖片 | 處理 | 原因 |
| --- | --- | --- | --- |
| 01_機器學習的類型 | [01_supervised_learning.png](01_機器學習的類型/images/01_supervised_learning.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |
| 01_機器學習的類型 | [02_unsupervised_learning.png](01_機器學習的類型/images/02_unsupervised_learning.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |
| 01_機器學習的類型 | [03_comparison.png](01_機器學習的類型/images/03_comparison.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 02_數據結構 | [01_features.png](02_數據結構/images/01_features.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |
| 02_數據結構 | [02_labels.png](02_數據結構/images/02_labels.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |
| 02_數據結構 | [03_data_structure_example.png](02_數據結構/images/03_data_structure_example.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 03_數據分割 | [01_train_test_split.png](03_數據分割/images/01_train_test_split.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 03_數據分割 | [02_three_way_split_validation.png](03_數據分割/images/02_three_way_split_validation.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 03_數據分割 | [03_split_workflow_and_overfitting.png](03_數據分割/images/03_split_workflow_and_overfitting.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 04_特徵工程 | [01_feature_selection.png](04_特徵工程/images/01_feature_selection.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 04_特徵工程 | [02_feature_scaling.png](04_特徵工程/images/02_feature_scaling.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |
| 04_特徵工程 | [03_categorical_encoding.png](04_特徵工程/images/03_categorical_encoding.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 05_模型參數 | [01_model_parameters.png](05_模型參數/images/01_model_parameters.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |
| 05_模型參數 | [02_hyperparameters.png](05_模型參數/images/02_hyperparameters.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 05_模型參數 | [03_parameters_vs_hyperparameters.png](05_模型參數/images/03_parameters_vs_hyperparameters.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 06_優化算法 | [01_gradient_descent.png](06_優化算法/images/01_gradient_descent.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |
| 06_優化算法 | [02_learning_rate.png](06_優化算法/images/02_learning_rate.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |
| 06_優化算法 | [03_batch_size_comparison.png](06_優化算法/images/03_batch_size_comparison.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 07_常見算法 | [01_knn.png](07_常見算法/images/01_knn.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |
| 07_常見算法 | [02_decision_tree.png](07_常見算法/images/02_decision_tree.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |
| 07_常見算法 | [03_svm.png](07_常見算法/images/03_svm.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |
| 07_常見算法 | [04_naive_bayes.png](07_常見算法/images/04_naive_bayes.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 08_集成學習 | [01_bagging.png](08_集成學習/images/01_bagging.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 08_集成學習 | [02_boosting.png](08_集成學習/images/02_boosting.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 08_集成學習 | [03_stacking.png](08_集成學習/images/03_stacking.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 09_評估指標 | [01_confusion_matrix.png](09_評估指標/images/01_confusion_matrix.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 09_評估指標 | [02_classification_metrics.png](09_評估指標/images/02_classification_metrics.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 09_評估指標 | [03_regression_metrics_errors.png](09_評估指標/images/03_regression_metrics_errors.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |
| 09_評估指標 | [04_r_squared.png](09_評估指標/images/04_r_squared.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |
| 10_模型性能問題 | [01_underfitting.png](10_模型性能問題/images/01_underfitting.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |
| 10_模型性能問題 | [02_overfitting.png](10_模型性能問題/images/02_overfitting.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |
| 10_模型性能問題 | [03_bias_variance_tradeoff.png](10_模型性能問題/images/03_bias_variance_tradeoff.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |
| 10_模型性能問題 | [04_learning_curves.png](10_模型性能問題/images/04_learning_curves.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |
| 10_模型性能問題 | [05_solutions_overview.png](10_模型性能問題/images/05_solutions_overview.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 11_模型泛化 | [01_generalization_concept.png](11_模型泛化/images/01_generalization_concept.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 11_模型泛化 | [02_generalization_gap.png](11_模型泛化/images/02_generalization_gap.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |
| 11_模型泛化 | [03_data_splitting.png](11_模型泛化/images/03_data_splitting.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 11_模型泛化 | [04_cross_validation.png](11_模型泛化/images/04_cross_validation.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 11_模型泛化 | [05_improve_generalization.png](11_模型泛化/images/05_improve_generalization.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 12_機器學習訓練過程 | [01_ml_lifecycle_pipeline.png](12_機器學習訓練過程/images/01_ml_lifecycle_pipeline.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 12_機器學習訓練過程 | [02_model_training_iteration_loop.png](12_機器學習訓練過程/images/02_model_training_iteration_loop.png) | 重製 SVG | 流程、表格或資訊卡適合以原生向量文字和線條重製。 |
| 12_機器學習訓練過程 | [03_underfitting_overfitting_spectrum.png](12_機器學習訓練過程/images/03_underfitting_overfitting_spectrum.png) | 保留 PNG | 包含散點、曲線、等高線或複雜座標；缺少原繪圖資料，描邊易使刻度與文字失真。 |

## 驗證

- 23 張 SVG 逐張於瀏覽器載入、截圖與視覺檢查。
- XML 解析、圖片連結存在性、未內嵌 PNG、保留原始檔檢查。
- 檢查 SVG 文字是否超出 viewBox。
- 重新產生 12 章網頁，檢查圖片載入與原有 42 個動畫場景掛載。

GitHub 的線上顯示需在提交、推送與部署後才能確認；本次沒有提交或發布。
