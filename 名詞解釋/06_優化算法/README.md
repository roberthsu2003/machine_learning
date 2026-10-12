# 6. 優化算法 (Optimization Algorithms)

<!-- interactive-glossary:start -->

🌐 **[開啟本章互動動畫教學](https://roberthsu2003.github.io/machine_learning/glossary/06-optimization/)** · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roberthsu2003/machine_learning/blob/main/%E5%90%8D%E8%A9%9E%E8%A7%A3%E9%87%8B/06_%E5%84%AA%E5%8C%96%E7%AE%97%E6%B3%95/README.ipynb) · 📓 **[實作筆記本](./README.ipynb)**

動畫場景：[梯度下降](https://roberthsu2003.github.io/machine_learning/glossary/06-optimization/#gradient-descent) · [學習率](https://roberthsu2003.github.io/machine_learning/glossary/06-optimization/#learning-rate) · [Batch／SGD／Mini-batch](https://roberthsu2003.github.io/machine_learning/glossary/06-optimization/#batch-size) · [優化器補充](https://roberthsu2003.github.io/machine_learning/glossary/06-optimization/#optimizers)

<!-- interactive-glossary:end -->

在機器學習中，模型一開始的參數通常是隨機初始化的（預測結果往往荒腔走板）。  
**優化算法（Optimizer）就是引導模型參數從「一無所知」走向「精準預測」的導航儀**，它的任務是沿著損失函數的地形，找到讓預測誤差最小化的一組最佳參數 $\theta^*$。

---

## 🎯 本章學習目標
1. 掌握**梯度下降法 (Gradient Descent)** 的數學直覺與幾何物理意義。
2. 搞懂**學習率 ($\alpha$)** 的步長效應，理解震盪與發散的原因。
3. 熟悉 **Batch Size（批次大小）** 的性能與泛化權衡，通曉常用優化器的演進脈絡。

---

## 1. 梯度下降法 (Gradient Descent)

### 核心定義
梯度下降法是一種基於一階導數的數值迭代優化算法。它計算損失函數 $J(\theta)$ 對參數 $\theta$ 的**梯度向量 $\nabla J(\theta)$**，並沿著**負梯度方向（局部最陡峭下降方向）**更新參數，步步逼近損失函數的極小值點。

### 💡 生活直觀比喻：濃霧中盲眼摸索下山
> 想像你被蒙上雙眼、置身於濃霧瀰漫的群山之中，任務是走到地勢最低的山谷底部。  
> 雖然你看不到全貌，但你可以用腳感受腳下地面的傾斜坡度（計算梯度）。  
> 只要**每一次都朝著腳下最陡的斜坡向下跨出一小步（沿負梯度前進）**，持續反覆這一步驟，最終就能抵達山谷盆地！

#### 📊 觀念圖解
> 🎬 互動動畫：[梯度下降](https://roberthsu2003.github.io/machine_learning/glossary/06-optimization/#gradient-descent)
![梯度下降法原理與變體對比](images/01_gradient_descent.png)

---

### 核心參數更新公式

$$\theta_{\text{new}} = \theta_{\text{old}} - \alpha \cdot \nabla J(\theta)$$

- $\theta$：模型參數（如權重 $w$ 與偏置 $b$）。
- $\nabla J(\theta)$：損失函數對參數的偏微分梯度（決定**前進方向**，數值為正表示向右上升，減去它即向左下降）。
- $\alpha$：**學習率 (Learning Rate)**，決定每次更新的**步伐大小 (Step Size)**。

---

### 梯度下降的三大實戰變體

| 變體名稱 | 每次更新使用的數據量 | 💡 比喻 | 優點 | 缺點 |
| :--- | :--- | :--- | :--- | :--- |
| **批次梯度下降 (Batch GD, BGD)** | **全部訓練樣本 (100%)** | 收集全校所有學生意見後才做決定 | 梯度估算精準，路徑平滑直達極值點。 | 當數據量龐大（數百萬筆）時，單步計算極其緩慢且耗盡記憶體。 |
| **隨機梯度下降 (Stochastic GD, SGD)** | **僅隨機抽取單一筆樣本 ($N=1$)** | 走在路上隨機抓一個路人問話就做決定 | 每步計算極快，具強烈隨機噪聲，容易跳出局部鞍點。 | 更新路徑劇烈劇烈震盪，難以平穩收斂在最低點。 |
| **小批次梯度下降 (Mini-Batch GD)** | **隨機抽取一小批樣本 (如 32, 64, 128)** | 每次抽樣一個小組（如 32 人）開會定案 | **兼具 BGD 的穩定度與 SGD 的高速度**，能充分利用 GPU 並行加速。 | 需要額外調整 Batch Size 超參數。當前深度學習絕對標準首選！ |

---

## 2. 學習率 (Learning Rate, $\alpha$)

學習率是優化算法中最核心、最敏感的超參數。

#### 📊 觀念圖解
> 🎬 互動動畫：[學習率](https://roberthsu2003.github.io/machine_learning/glossary/06-optimization/#learning-rate)
![學習率大小對訓練收斂的影響](images/02_learning_rate.png)

### 步長大小引發的性能光譜

![學習率大小引發的性能光譜](images/02_learning_rate_spectrum.svg)

### 實務調優策略
1. **典型初始值**：常見的嘗試起點為 `0.001`、`0.01`。
2. **學習率衰減 (Learning Rate Decay / Scheduling)**：
   - 訓練初期（離谷底尚遠）：使用較大學習率大步快跑。
   - 訓練後期（接近最優點）：動態調小學習率（如 Cosine Annealing），細緻落入谷底最深處。

---

## 3. 批次大小 (Batch Size)

#### 📊 觀念圖解
> 🎬 互動動畫：[Batch／SGD／Mini-batch](https://roberthsu2003.github.io/machine_learning/glossary/06-optimization/#batch-size)
![批次大小權衡與民調抽樣比喻](images/03_batch_size_comparison.svg)

[查看原始 PNG 圖片](images/03_batch_size_comparison.png)

### 💡 直觀理解：民調抽樣的比喻
- **大批次（如 512, 1024 / 大規模民調）**：問話人次多，誤差小，決策方向穩如泰山；但耗費大量記憶體 (VRAM)。有時過於「拘謹」，容易跌入尖銳的局部不良極值（泛化稍差）。
- **小批次（如 16, 32 / 街頭隨機抽問幾人）**：問話速度極快，記憶體佔用極低。雖然每次更新的方向有些搖晃震盪，但**適度的噪聲反而能將模型震出平庸的淺坑（鞍點），尋找到更寬廣、泛化更好的全局平坦最優谷底**！

### 🔄 訓練計量三劍客：Batch Size、Step (Iteration) 與 Epoch

在觀察模型訓練或動畫演示時，很多人常有疑問：  
**「如果資料集只有 32 筆，我設批次為 8，為什麼更新次數會走 20 次？資料不是讀完 32 筆就結束了嗎？」**

答案是：**資料不是只讀一次，而是會被重新洗牌（Shuffle）、反覆研讀許多輪！**  
為了精確描述訓練進度，我們必須釐清以下三個核心概念：

| 術語 | 角色定義 | 在「32 筆資料、Batch=8」的例子中 |
| :--- | :--- | :--- |
| **Batch Size (批次大小)** | 每次邁出一步（更新一次參數）所消耗的樣本數量。 | **8 筆** |
| **Step / Iteration (步數/迭代)** | 模型「更新一次參數」的計數單位（每讀完一個 Batch 就是 1 個 Step）。 | **走 1 步**（累積完成了 20 步） |
| **Epoch (訓練輪次)** | **全部訓練資料被模型完整看過一遍**的週期。 | $32 \div 8 =$ **每 4 步算 1 個 Epoch** |

#### 💡 生活比喻：考前刷題庫
- **總題庫有 32 題**。你每次從中抽出 **8 題** 一起寫（Batch Size = 8）。
- 你寫完 8 題對一次答案、檢討改進一次（走 1 個 Step）。
- 當你做了 4 次（4 個 Steps），剛好把這 32 題都寫過一遍，這就完成了 **第 1 輪複習 (1 個 Epoch)**。
- **但只做一遍就能考滿分嗎？通常不能！** 模型只看一遍資料根本記不住規律，因此必須把題目**重新打散順序（Shuffle）**，繼續刷第 2 輪、第 3 輪……當完成 20 次更新時，相當於把這份題庫**反覆精讀了整整 5 輪（5 個 Epochs）**！

#### 📐 三者的數學連動關係
$$\text{總更新步數 (Steps)} = \text{總輪數 (Epochs)} \times \left( \frac{\text{資料集總樣本數}}{\text{批次大小 (Batch Size)}} \right)$$

> [!TIP]
> **GPU 硬體加速法則**：  
> 在配置 Batch Size 時，幾乎總是設定為 **2 的次方數**（如 16, 32, 64, 128, 256, 512）。這是因為 GPU 的硬體計算單元（CUDA Cores / Tensor Cores）在 2 的次方記憶體對齊下吞吐效率最高！

### 💡 實務思考：學習率與 Batch Size 要一起調嗎？不同的演算法超參數有何不同？

很多初學者會疑惑：**「訓練模型時，只要調學習率就好，還是連 Batch Size 也要一起調？不同的演算法調整方式一樣嗎？」**

答案是：**它們通常需要成對調整，而且不同演算法能調整的超參數也不一樣！**

1. **學習率與 Batch Size 是「搭檔關係」**：
   - 實務上調整模型時，這兩者往往牽一髮而動全身。
   - 一般來說，若把 **Batch Size 調大**，每次計算的方向更穩、但整個 Epoch 裡的更新次數變少了，通常會把**學習率也相應調大**以維持訓練效率；反之，若 **Batch Size 調小**，更新震盪較大，學習率就必須設定得相對保守。

2. **演算法不同，需要調的超參數也完全不同**：
   每種優化算法提供的「控制旋鈕」各不相同：

| 優化演算法 | 實務上需要調整的超參數 | 特點說明 |
| :--- | :--- | :--- |
| **BGD (全部數據)** | 學習率 ($\alpha$) | **不用調 Batch Size**（因為永遠是 100% 數據），只需專注調學習率。 |
| **SGD ($N=1$)** | 學習率 ($\alpha$) | Batch Size 固定為 1，噪聲大，需特別注意學習率不能太大。 |
| **Mini-Batch GD** | **學習率 ($\alpha$) + 批次大小 (Batch Size)** | 現代深度學習標準搭配，通常需要**同時搜尋這兩者的最佳組合**。 |
| **Momentum (動量法)** | 學習率 + Batch Size + **動量係數 ($\beta$)** | 多了一個「慣性」超參數（通常預設為 0.9）。 |
| **Adam / 自適應演算法** | 學習率 + Batch Size（內部自適應參數通常先採預設值） | 演算法會自己動態微調各參數步長，但**起始學習率與批次大小**依然是關鍵。 |

---

## 4. 拓展視野：現代優化器家族演進

簡單來說，**優化器（Optimizer）就是訓練時「自動幫你調整模型參數」的核心引擎**！模型算出預測誤差後，工程師不需要手動修改權重數值，全靠優化器自動計算方向並把參數更新到位。

不過，不同的優化器在「自動調整」時的聰明程度大不相同——傳統方法（如 SGD）調整時步伐死板固執，而現代優化器則學會了自帶慣性與自適應調速：

![現代優化器家族演進脈絡](images/04_optimizers_evolution.svg)

- **為什麼大家愛用 Adam？**：  
  Adam (Adaptive Moment Estimation) 就是最受歡迎的「全自動自適應優化器」！它同時具備「慣性衝力（避免卡在鞍點）」與「自動調整步長（劇烈變動的特徵自動小步走、罕見變動的特徵自動大步走）」，是目前大多數機器學習與深度學習任務中表現最魯棒、最省心的默認選擇！

---

## 📌 本章精華速記
1. **梯度下降**沿著損失函數最陡峭的反方向下山，推動參數持續走向最小誤差。
2. **學習率**決定前進步長；太小收斂牛步，太大震盪發散，適度配合衰減策略效果最佳。
3. **小批次 (Mini-batch)** 是現代工業界標準，兼顧計算效率、GPU 平行吞吐與噪聲泛化優勢。
4. **優化器就是自動幫你調整參數的引擎**；現代專案若不知從何選起，首選 **Adam 優化器** 作為穩健的強力起步基準（它連步長都會智慧自適應微調）。

---

[⏮️ 上一章：05_模型參數](../05_模型參數/README.md) | [📑 返回目錄](../README.md) | [⏭️ 下一章：07_常見算法](../07_常見算法/README.md)
