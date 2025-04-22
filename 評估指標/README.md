# 機器學習常用的評估指標

在機器學習中，評估指標是用來衡量模型性能的工具。不同的任務類型需要不同的評估指標。以下是一些常用的評估指標，並根據任務類型進行分類。

## 分類任務

## 分類任務

*   **準確率 (Accuracy)**：正確分類的樣本數佔總樣本數的比例。
    *   適用於類別分佈均衡的情況。
    *   公式：`Accuracy = (TP + TN) / (TP + TN + FP + FN)`
    *   [範例](accuracy.ipynb)
*   **精確率 (Precision)**：在所有被預測為正例的樣本中，實際為正例的比例。
    *   關注預測為正例的準確性。
    *   公式：`Precision = TP / (TP + FP)`
    *   [範例](precision.ipynb)
*   **召回率 (Recall)**：在所有實際為正例的樣本中，被正確預測為正例的比例。
    *   關注正例的覆蓋程度。
    *   公式：`Recall = TP / (TP + FN)`
    *   [範例](recall.ipynb)
*   **F1 分數 (F1-score)**：精確率和召回率的調和平均數。
    *   綜合考慮精確率和召回率。
    *   公式：`F1-score = 2 * (Precision * Recall) / (Precision + Recall)`
    *   [範例](f1_score.ipynb)
*   **AUC-ROC 曲線 (AUC-ROC Curve)**：ROC 曲線下的面積，用於評估二元分類器的性能。
    *   AUC 值越大，模型性能越好。
    *   [範例](auc_roc_curve.ipynb)
*   **混淆矩陣 (Confusion Matrix)**：用於可視化分類模型的性能，顯示 TP、TN、FP、FN 的數量。
    *   [範例](confusion_matrix.ipynb)

## 迴歸任務

*   **平均絕對誤差 (MAE)**：預測值與實際值之間絕對誤差的平均值。
    *   公式：`MAE = (1/n) * Σ|y_i - ŷ_i|`
    *   [範例](mae.ipynb)
*   **均方誤差 (MSE)**：預測值與實際值之間差的平方的平均值。
    *   公式：`MSE = (1/n) * Σ(y_i - ŷ_i)^2`
    *   [範例](mse.ipynb)
*   **均方根誤差 (RMSE)**：均方誤差的平方根。
    *   公式：`RMSE = √(MSE)`
    *   [範例](rmse.ipynb)
*   **R 平方 (R-squared)**：衡量模型解釋數據變異性的程度。
    *   值越接近 1，模型擬合效果越好。
    *   [範例](r_squared.ipynb)
