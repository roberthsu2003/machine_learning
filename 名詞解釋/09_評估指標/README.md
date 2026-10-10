# 9. 評估指標 (Performance Metrics)

<!-- interactive-glossary:start -->

🌐 **[開啟本章互動動畫教學](https://roberthsu2003.github.io/machine_learning/glossary/09-evaluation-metrics/)**

動畫場景：[混淆矩陣](https://roberthsu2003.github.io/machine_learning/glossary/09-evaluation-metrics/#confusion-matrix) · [分類指標與 ROC](https://roberthsu2003.github.io/machine_learning/glossary/09-evaluation-metrics/#classification-metrics) · [MAE、MSE、RMSE](https://roberthsu2003.github.io/machine_learning/glossary/09-evaluation-metrics/#regression-errors) · [R²](https://roberthsu2003.github.io/machine_learning/glossary/09-evaluation-metrics/#r-squared)

<!-- interactive-glossary:end -->

在機器學習專案中，**評估指標（Evaluation Metrics）是用來客觀衡量模型品質、診斷學習盲點與推動模型調優的指南針**。

不同的任務型態（分類 vs 回歸）以及不同的商業場景代價（如漏報代價 vs 誤報代價），必須選用相應的指標。如果選錯了指標，即使模型分數看似高達 99%，在真實業務落地時也可能帶來災難性的虧損！

---

## 🎯 本章學習目標
1. 徹底理解**混淆矩陣 (Confusion Matrix)** 的四象限組成與雙字母記憶口訣。
2. 搞懂分類四大指標（Accuracy, Precision, Recall, F1）的核心關注點與取捨權衡（Tradeoff）。
3. 掌握回歸任務的核心量化指標（MAE, MSE, RMSE, $R^2$）的幾何意義與單位特點。

---

## 1. 分類評估指標 (Classification Metrics)

### 混淆矩陣 (Confusion Matrix)

混淆矩陣是一個二維交叉統計表，精確總結了分類模型的預測結果與真實情況，是所有分類衍生指標的基石。

#### 📊 觀念圖解
> 🎬 互動動畫：[混淆矩陣](https://roberthsu2003.github.io/machine_learning/glossary/09-evaluation-metrics/#confusion-matrix)
![二元混淆矩陣結構與四象限圖解](images/01_confusion_matrix.png)

### 四象限組成概念

| 象限名詞 | 英文全稱 | 真實情況 | 模型預測 | 💡 生活實例（醫療檢測） |
| :--- | :--- | :---: | :---: | :--- |
| **真正例 (TP)** | True Positive | **陽性 (+)** | **陽性 (+)** | 病患確實染病，檢測結果精確抓出為陽性（成功確診）。 |
| **真負例 (TN)** | True Negative | **陰性 (-)** | **陰性 (-)** | 受檢者完全健康，檢測結果回報為陰性（平安無事）。 |
| **假正例 (FP)** | False Positive<br>*(Type I Error 第一型錯誤 / 誤報)* | **陰性 (-)** | **陽性 (+)** | 受檢者明明健康，檢測卻誤判為陽性（虛驚一場、冤枉好人）。 |
| **假負例 (FN)** | False Negative<br>*(Type II Error 第二型錯誤 / 漏報)* | **陽性 (+)** | **陰性 (-)** | 病患確實染病，檢測卻漏判為陰性（**致命漏網之魚！**）。 |

> 💡 **雙字母記憶口訣**：
> - **第一個字母 (True / False)**：代表模型「**猜得對不對**」（T = 猜對，F = 猜錯）。
> - **第二個字母 (Positive / Negative)**：代表模型「**預測它是哪一類**」（P = 猜正類，N = 猜負類）。

---

### 分類四大衍生指標體系

#### 📊 觀念圖解
> 🎬 互動動畫：[分類指標與 ROC](https://roberthsu2003.github.io/machine_learning/glossary/09-evaluation-metrics/#classification-metrics)
![分類指標體系：公式、適用場景與權衡關係](images/02_classification_metrics.png)

---

### 1. 準確率 (Accuracy)
- **數學公式**：
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
- **含義**：所有樣本中，被模型精確猜對的總比例。
- **🚨 致命陷阱：類別不平衡盲點**  
  > 假設某種罕見癌症在人口中患病率為 0.1%（1,000 人中只有 1 人得病）。  
  > 一個完全沒學習能力的「無腦模型」，不管誰來都全猜「健康 (Negative)」，其準確率依然高達 **99.9%**！但此模型對抓出病患毫無貢獻。因此**面對不平衡數據，絕對不能只看 Accuracy**！

---

### 2. 精確率 (Precision / 查準率)
- **數學公式**：
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **含義**：在所有**被模型猜測為正類**的樣本中，實際上真正為正類的比例。
- **關注焦點**：**極力消滅 FP（誤報）**！寧可少猜，也絕不能冤枉好人。
- **經典應用場景**：
  - **垃圾郵件過濾**：把重要工作通知或銀行驗證碼信件當垃圾郵件丟掉（FP）的代價無法承受。
  - **法庭宣判定罪**：寧可放過嫌疑犯，絕不可冤枉無辜好人入獄。
  - **精準電商推薦**：推薦清單必須精準切中用戶喜好，避免打擾顧客。

---

### 3. 召回率 (Recall / 查全率 / Sensitivity)
- **數學公式**：
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **含義**：在所有**真實為正類**的樣本中，被模型成功抓出的比例。
- **關注焦點**：**極力消滅 FN（漏報）**！寧可抓錯十個，也絕不能漏放一個。
- **經典應用場景**：
  - **惡性腫瘤篩檢**：若病患得癌卻被誤判為健康（FN），將延誤就醫導致喪命！
  - **信用卡盜刷風控**：寧可暫時凍結可疑交易打電話確認，也絕不讓盜刷成功。
  - **防空預警雷達**：寧可把大批海鳥誤判為敵機，也絕不能漏掉一枚真實飛彈。

---

### 4. F1 分數 (F1-Score)
- **數學公式**：
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$
- **含義**：精確率與召回率的**調和平均數 (Harmonic Mean)**。
- **為什麼用調和平均，不用算術平均？**  
  調和平均數會對極端微小值施加嚴厲懲罰。如果 Precision 高達 99% 但 Recall 只有 1%，其算術平均為 50%，但 F1 分數會直接跌落至接近 2%！只有在兩者皆優異時，F1 才會高。
- **適用場景**：數據不平衡，且業務上同時需要兼顧準確性與覆蓋面的綜合評估。

---

### 5. ROC-AUC (受試者工作特徵曲線與曲線下面積)

在實務中，分類模型輸出的往往是「屬於正類的機率值」（如 $0.78$）。我們通常以 $0.5$ 作為門檻（Threshold），大於 $0.5$ 判為正類。但如果把門檻調低（如 $0.2$）或調高（如 $0.8$），Precision 與 Recall 就會此消彼長。

- **ROC 曲線**：以假正例率 (FPR) 為橫軸、真正例率 (TPR / Recall) 為縱軸，描繪門檻從 1 變動到 0 時模型的表現軌跡。
- **AUC (Area Under Curve)**：ROC 曲線下方的面積（介於 0.5 到 1.0 之間）。
  - $\text{AUC} = 1.0$：完美無瑕的頂級模型。
  - $\text{AUC} = 0.5$：等同於盲猜拋硬幣。
  - **最大優點**：**完全不受類別不平衡干擾**，且能客觀評價模型跨越所有可能門檻的綜合排序能力！

---

## 2. 回歸評估指標 (Regression Metrics)

回歸任務的輸出為連續數值（如房價、氣溫、營收）。我們透過評估真實值 $y$ 與預測值 $\hat{y}$ 之間的殘差來衡量擬合品質。

#### 📊 觀念圖解
> 🎬 互動動畫：[MAE、MSE、RMSE](https://roberthsu2003.github.io/machine_learning/glossary/09-evaluation-metrics/#regression-errors)
![回歸誤差三大指標對比](images/03_regression_metrics_errors.png)

---

### 三大誤差指標對照

| 指標名稱 | 數學公式 | 單位尺度 | 對極端離群值敏感度 | 特性與適用場景 |
| :--- | :---: | :---: | :---: | :--- |
| **MAE**<br>(平均絕對誤差) | $$\frac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$$ | 原始單位<br>(如: 萬元) | **低 (Robust)**<br>線性權重，抗干擾強 | 最直觀好懂（平均猜錯多少錢），適合數據中存在偶發異常雜訊時使用。 |
| **MSE**<br>(均方誤差) | $$\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$ | 平方單位<br>(如: 萬元$^2$) | **極高**<br>平方放大極端大誤差 | 數學性質極佳（處處可導），是梯度下降優化器最經典的損失函數；但缺乏物理直觀性。 |
| **RMSE**<br>(均方根誤差) | $$\sqrt{\text{MSE}}$$ | 原始單位<br>(如: 萬元) | **高**<br>兼具平方重罰懲罰性 | **對 MSE 開根號還原回原始物理尺度**，同時保留對大誤差的懲罰，工業界競賽最愛指標！ |

---

### 決定係數 ($R^2$ 分數 / R-Squared)

#### 核心定義
$R^2$ 衡量回歸模型**「能夠解釋數據中總變異性 (Variance) 的百分比」**。它是無量綱（無單位）的標準化指標，以「直接猜母體平均數 $\bar{y}$ 的基準笨模型」為對照組。

#### 📊 觀念圖解
> 🎬 互動動畫：[R²](https://roberthsu2003.github.io/machine_learning/glossary/09-evaluation-metrics/#r-squared)
![決定係數 R² 原理：衡量模型解釋資料變異的能力](images/04_r_squared.png)

#### 數學公式

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

- $SS_{\text{res}}$（殘差平方和）：模型沒能解釋的預測殘差平方和。
- $SS_{\text{tot}}$（總平方和）：數據本身的原始總波動平方和。

#### $R^2$ 數值解讀光譜
- **$R^2 = 1.0$ (100%)**：完美預測！所有樣本點分毫不差落在回歸線上。
- **$0 < R^2 < 1.0$**：一般正常模型（如 $R^2 = 0.85$ 表示模型解釋了數據中 85% 的波動）。
- **$R^2 = 0.0$**：等同瞎猜。模型水平跟直接猜平均值 $\bar{y}$ 完全一樣。
- **$R^2 < 0.0$ (負數！)**：模型預測誤差比直接猜均值還要離譜，說明模型架構設計存在重大錯誤。

---

## 📌 本章精華速記
1. **分類評估**：
   - 類別不平衡千萬別看 Accuracy。
   - 怕誤報（冤枉好人）看 **Precision**；怕漏報（漏網之魚）看 **Recall**。
   - 綜合平衡選 **F1-Score**；門檻無關排序能力選 **ROC-AUC**。
2. **回歸評估**：
   - 講求直觀抗噪選 **MAE**；懲罰大誤差且具備物理單位選 **RMSE**。
   - 跨專案橫向比較擬合能力選無量綱的 **$R^2$ 分數**。

---

[⏮️ 上一章：08_集成學習](../08_集成學習/README.md) | [📑 返回目錄](../README.md) | [⏭️ 下一章：10_模型性能問題](../10_模型性能問題/README.md)
