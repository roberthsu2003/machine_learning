from pathlib import Path
from textwrap import dedent
import json
import shutil
from urllib.parse import quote

ROOT = Path("AI提問與機器學習實戰")
BOOK = Path("資料科學自學聖經")
RAW = "https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E6%8F%90%E5%95%8F%E8%88%87%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E5%AF%A6%E6%88%B0"
SPECS = [
 ("04_葡萄酒品種辨識_經典分類演算法篇", "葡萄酒品種辨識_經典分類演算法篇.ipynb", "🍷 範例 04：葡萄酒品種辨識（經典分類演算法篇）", "wine.csv", BOOK/"03_分類模型數據集/wine.csv", "酒種類", "多類別分類", "K-近鄰、決策樹、核化 SVM、高斯樸素貝氏、梯度提升與 Stacking", "classic"),
 ("05_乳癌診斷_PCA降維篇", "乳癌診斷_PCA降維篇.ipynb", "🩺 範例 05：乳癌診斷與 PCA 降維（降維篇）", "breastCancer.csv", BOOK/"03_分類模型數據集/breastCancer.csv", "種類", "二元分類＋降維", "PCA 主成分分析、邏輯迴歸與核化 SVM", "pca"),
 ("06_商場客群輪廓探索_進階分群篇", "商場客群輪廓探索_進階分群篇.ipynb", "🧭 範例 06：商場客群輪廓探索（進階分群篇）", "customer.csv", BOOK/"04_集群與關聯規則/customer.csv", "", "非監督式分群", "階層式集群與 DBSCAN 密度分群", "cluster"),
 ("07_葡萄酒品種辨識_神經網路篇", "葡萄酒品種辨識_神經網路篇.ipynb", "🧠 範例 07：葡萄酒品種辨識（神經網路篇）", "wine.csv", BOOK/"03_分類模型數據集/wine.csv", "酒種類", "多類別分類", "多層感知器 MLP 神經網路與邏輯迴歸基準線", "mlp"),
]

def L(text): return [x + "\n" for x in dedent(text).strip().splitlines()]
def M(text): return {"cell_type":"markdown","metadata":{},"source":L(text)}
def C(text): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":L(text)}
def guide(step, note):
 return M(f"""#### 💬【{step} 學習指引與 AI 提問 Prompt】

{note}

> ❓ 請用生活化比喻解釋這一步的原理，並指出一個常見誤解。

> 🛠️ 請只改一個超參數，比較修改前後的結果；不要使用測試集選參數。""")

ENV = """# [步驟 0 程式碼] 載入套件與設定繁體中文字型
import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import mplfonts
    from matplotlib import font_manager
    from mplfonts import use_font
    for f in glob.glob(os.path.join(os.path.dirname(mplfonts.__file__), "fonts", "*.[ot]tf")):
        font_manager.fontManager.addfont(f)
    use_font("Noto Sans CJK TC")
except Exception:
    print("提示：中文字型設定失敗時，模型仍可正常執行。")
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")"""

COLAB_SETUP = """# 只在 Colab 缺少套件時安裝；本機使用專案的 requirements.txt
try:
    import google.colab
except ImportError:
    pass
else:
    import importlib.util
    import subprocess
    import sys
    missing = [name for name in ("mplfonts", "seaborn", "scipy") if importlib.util.find_spec(name) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])"""

def classified(target, kind):
 base = f"""from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report
X = df.drop(columns="{target}")
y = df["{target}"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"""
 if kind == "classic":
  models = """from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
models = {
 "KNN": Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(5))]),
 "決策樹": DecisionTreeClassifier(max_depth=4, random_state=42),
 "核化 SVM": Pipeline([("scale", StandardScaler()), ("model", SVC(C=2, kernel="rbf"))]),
 "高斯貝氏": GaussianNB(),
 "梯度提升": GradientBoostingClassifier(random_state=42),
 "Stacking": StackingClassifier(estimators=[
  ("knn", Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(5))])),
  ("tree", DecisionTreeClassifier(max_depth=4, random_state=42)), ("nb", GaussianNB())],
  final_estimator=LogisticRegression(max_iter=2000), cv=5)}"""
 elif kind == "pca":
  models = """from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
models = {
 "邏輯迴歸基準線": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000))]),
 "PCA + 核化 SVM": Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=.95, random_state=42)), ("model", SVC(C=1, kernel="rbf"))])}"""
 else:
  models = """from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
models = {
 "邏輯迴歸基準線": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000))]),
 "MLP 神經網路": Pipeline([("scale", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(32,), early_stopping=True, validation_fraction=.2, n_iter_no_change=30, learning_rate_init=.005, max_iter=3000, random_state=42))])}"""
 diagnostic = """# [步驟 5 程式碼] PCA 降維診斷與二維視覺化
pca_probe = PCA().fit(StandardScaler().fit_transform(X_train))
cumulative = np.cumsum(pca_probe.explained_variance_ratio_)
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(cumulative)+1), cumulative, marker="o")
plt.axhline(.95, color="crimson", linestyle="--", label="95% 資訊保留")
plt.xlabel("主成分數"); plt.ylabel("累積解釋變異比例"); plt.legend(); plt.show()

pca_2d = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X_train))
plt.figure(figsize=(7, 5))
plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=y_train, cmap="Set1", alpha=.75)
plt.title("乳癌資料 PCA 二維投影（顏色為原始類別）"); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.show()""" if kind == "pca" else """# [步驟 5 程式碼] 模型診斷圖
plt.figure(figsize=(8, 4))
result_df["5 折 Accuracy"].sort_values().plot(kind="barh", color="steelblue")
plt.xlabel("5 折交叉驗證 Accuracy"); plt.title("模型選型比較：不要只看單一次切分"); plt.show()"""
 if kind == "mlp":
  diagnostic = """# [步驟 5 程式碼] MLP 訓練損失曲線
mlp_for_curve = models["MLP 神經網路"].fit(X_train, y_train)
plt.figure(figsize=(8, 4))
plt.plot(mlp_for_curve.named_steps["model"].loss_curve_)
plt.title("MLP 訓練損失曲線"); plt.xlabel("迭代次數"); plt.ylabel("Loss"); plt.show()
print("Early Stopping 會在驗證表現不再改善時提早停止，降低過度擬合風險。")"""
 return [
  M("---\n### 【步驟 2】：特徵關係與視覺化\n先看類別分布及兩個特徵的關係；這是探索資料，不是用兩個特徵取代完整模型。"),
  C(f"""# [步驟 2 程式碼] 類別分布與前兩個特徵
X = df.drop(columns="{target}")
display(df["{target}"].value_counts().sort_index().rename("筆數"))
cols = list(X.columns)
sns.scatterplot(data=df, x=cols[0], y=cols[1], hue="{target}", palette="Set2", s=65)
plt.title("前兩個特徵的類別分布"); plt.show()"""),
  guide("步驟 2", "散佈圖只能同時呈現兩個特徵；真正模型仍使用全部特徵。類別分布也會影響 Accuracy 的解讀。"),
  M("---\n### 【步驟 3】：資料切分、Pipeline 與交叉驗證\n所有前處理都放進 Pipeline，並只在訓練資料中擬合，避免測試資料洩漏。"),
  C(base + "\n" + models + """
# [步驟 3 程式碼] Pipeline 模型比較與 5 折交叉驗證
cv_results = {name: cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean() for name, model in models.items()}
result_df = pd.Series(cv_results, name="5 折 Accuracy").sort_values(ascending=False).to_frame()
display(result_df.style.format("{:.3f}"))"""),
  guide("步驟 3", "交叉驗證只在訓練資料內完成。KNN、SVM 和神經網路都需要標準化；樹模型通常不需要。"),
  M("---\n### 【步驟 4】：測試集大考與指標解讀\n選定模型後才碰測試集。它的角色是最終驗收，不是反覆調參用的練習題。"),
  C("""# [步驟 4 程式碼] 測試集大考
best_name = result_df.index[0]
best_model = models[best_name].fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print(f"交叉驗證最佳模型：{best_name}")
print(f"測試集 Accuracy：{accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, digits=3))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title(f"{best_name}：測試集混淆矩陣"); plt.show()"""),
  guide("步驟 4", "高 Accuracy 不代表唯一最佳模型，也必須考慮可解釋性、速度與維護成本。"),
  M("---\n### 【步驟 5】：模型診斷與可解釋性\n利用圖表了解模型為什麼這樣表現，再決定是否值得投入更多複雜度。"),
  C(diagnostic),
  guide("步驟 5", "診斷圖用來提出下一個問題，不是保證模型一定能用。請留意資料量、錯誤成本與公平性。"),
  M("---\n### 【步驟 6】：未見資料的實務推論\n最後以一筆模型從未參與訓練的資料模擬部署時的預測流程。"),
  C("""# [步驟 6 程式碼] 一筆未見資料的推論
new_case = X_test.iloc[[0]]
print("模型預測：", best_model.predict(new_case)[0])
print("實際標籤（課後核對）：", y_test.iloc[0])"""),
  guide("步驟 6", "這筆資料在模型訓練時未出現；請先猜測再看模型答案。真實應用還需要持續監控資料漂移。")]

def clustered():
 return [
  M("---\n### 【步驟 2】：選擇客群特徵與標準化\n距離型演算法會受數值尺度影響；先將年齡、收入和消費指數改用同一把尺。"),
  C("""# [步驟 2 程式碼] 特徵選擇與標準化
from sklearn.preprocessing import StandardScaler
features = ["年齡", "收入(千)", "消費指數(1-100)"]
X = df[features]
display(X.describe().T)
X_scaled = StandardScaler().fit_transform(X)"""),
  guide("步驟 2", "收入與消費指數尺度不同；距離型演算法必須先用同一把尺。"),
  M("---\n### 【步驟 3】：階層式集群與樹狀圖\n從每位顧客各自成群開始逐步合併，利用樹狀圖觀察不同群數的可能性。"),
  C("""# [步驟 3 程式碼] 階層式集群與樹狀圖
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
linked = linkage(X_scaled, method="ward")
plt.figure(figsize=(14,5)); dendrogram(linked, truncate_mode="lastp", p=25)
plt.title("階層式集群樹狀圖"); plt.ylabel("合併距離"); plt.show()
df["階層式群組"] = AgglomerativeClustering(n_clusters=4).fit_predict(X_scaled)
display(df.groupby("階層式群組")[features].mean().round(1))"""),
  guide("步驟 3", "階層式集群由每人一群開始合併；樹狀圖切線的高度決定群數。"),
  M("---\n### 【步驟 4】：DBSCAN 密度分群與雜訊\nDBSCAN 不必先指定群數，但要用資料密度選擇 `eps` 與 `min_samples`。"),
  C("""# [步驟 4 程式碼] DBSCAN 找密集客群與雜訊
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
distances, _ = NearestNeighbors(n_neighbors=5).fit(X_scaled).kneighbors(X_scaled)
plt.plot(np.sort(distances[:,-1])); plt.title("第 5 近鄰距離圖"); plt.show()
df["DBSCAN群組"] = DBSCAN(eps=.75, min_samples=5).fit_predict(X_scaled)
print("雜訊（-1）筆數：", (df["DBSCAN群組"] == -1).sum())
sns.scatterplot(data=df, x="收入(千)", y="消費指數(1-100)", hue="DBSCAN群組", palette="tab10", s=70)
plt.title("DBSCAN 分群結果"); plt.show()"""),
  guide("步驟 4", "DBSCAN 的 -1 是雜訊，不是第負一群；試著比較 eps 太小或太大時的結果。"),
  M("---\n### 【步驟 5】：客群輪廓與方法限制\n群組編號沒有大小意義；請用群內平均特徵與人數為每群命名、提出行銷策略。"),
  C("""# [步驟 5 程式碼] 客群輪廓與方法限制
display(df.groupby("階層式群組")[features].agg(["mean","count"]).round(1))
print("階層式集群與 DBSCAN 沒有原生 predict；新客戶不能直接套用舊標籤。")"""),
  guide("步驟 5", "群組編號沒有高低順序，請以各群平均輪廓與人數提出行銷建議。"),
  M("---\n### 【步驟 6】：實務決策模擬\n抽一位既有顧客，根據其所在群組提出一個可驗證的行銷假設。"),
  C("""# [步驟 6 程式碼] 隨機抽取既有顧客的群組建議
case = df.sample(1, random_state=42)
display(case[features + ["階層式群組", "DBSCAN群組"]])
print("請先以收入、消費指數與年齡描述此顧客，再提出一項可 A/B 測試的行銷策略。")"""),
  guide("步驟 6", "分群是提出假設的工具，不是自動化的行銷決策。驗證策略時仍應設計對照組與成功指標。")]

def notebook(spec):
 folder, file, title, dataset, source, target, task, algos, kind = spec
 target_note = f"目標欄位：{target}" if target else "沒有標籤，目標是探索客群結構"
 body = clustered() if kind == "cluster" else classified(target, kind)
 cells = [M(f'<a href="https://colab.research.google.com/github/roberthsu2003/machine_learning/blob/main/AI%E6%8F%90%E5%95%8F%E8%88%87%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E5%AF%A6%E6%88%B0/{quote(folder)}/{quote(file)}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'),
  C(COLAB_SETUP),
  M(f"""# {title} — AI 伴學實戰筆記本

歡迎來到本實戰筆記本！本筆記本沿用範例 01 的 **「一段程式碼 + 一段 AI 互動學習指引」** 架構：

- 每個程式儲存格都提供可直接執行的資料處理、建模與評估流程。
- 每段程式碼的正下方都有白話說明與專屬 Prompt；可直接複製給 AI 追問原理或練習調整。
- 資料取自《資料科學自學聖經》的 `{dataset}`，在 Colab 找不到檔案時會自動下載。

請從步驟 0 依序執行。交叉驗證與調參只能使用訓練資料；測試集保留到最後一次驗收。"""),
  M("---  \n### 【步驟 0】：環境準備與繁體中文字型設定\n載入資料科學套件並設定中文字型，避免圖表顯示亂碼。"), C(ENV),
  guide("步驟 0", "載入資料處理與視覺化套件後，先確認環境再開始模型實驗；中文字型失敗不會影響模型訓練。"),
  M("---  \n### 【步驟 1】：資料載入與結構探索（EDA）\n讀取資料、查看欄位型態與摘要統計，先了解資料再下模型。"),
  C(f"""# [步驟 1 程式碼] 資料載入與結構探索
if not os.path.exists("{dataset}"):
    from urllib.request import urlretrieve
    urlretrieve("{RAW}/{quote(folder)}/{quote(dataset)}", "{dataset}")
df = pd.read_csv("{dataset}")
print(f"資料筆數：{{len(df):,}}；欄位數：{{df.shape[1]}}")
display(df.head())
display(df.describe(include="all").T)
print("{task}；{target_note}")"""),
  guide("步驟 1", f"本專案要學會：{algos}。先檢查資料型態、缺失值、重複值與可疑欄位。")] + body
 return {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}

def readme(spec):
 folder, file, title, dataset, source, target, task, algos, kind = spec
 y = target or "無標籤（探索群體）"
 return dedent(f"""# {title}

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roberthsu2003/machine_learning/blob/main/AI%E6%8F%90%E5%95%8F%E8%88%87%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E5%AF%A6%E6%88%B0/{folder}/{file})

本範例以《資料科學自學聖經》的 **{dataset}** 為資料來源，沿用範例 01 的完整 AI 伴學教學架構：穩定程式碼、逐段白話解說、可複製的 Prompt、交叉驗證與獨立桌面 GUI。

---

## 🎯 你將學會的核心觀念與專案學習地圖

| 核心學習觀念 | 重要觀念與實務重點 | 📖 專案名詞解釋（點擊研讀） | 🤖 專案模型與實戰主題 |
| :--- | :--- | :--- | :--- |
| 問題定義 | 辨識 {task}，明確區分特徵 X 與目標 y。 | [機器學習類型](../../名詞解釋/01_機器學習的類型/README.md) | 目標：{y} |
| 資料與前處理 | 檢查型態、尺度、缺失值與資料品質；前處理只能在訓練資料 fit。 | [特徵工程](../../名詞解釋/04_特徵工程/README.md) | EDA、標準化、Pipeline |
| 演算法選型 | 實作與比較：{algos}。 | [常見算法](../../名詞解釋/07_常見算法/README.md) | 交叉驗證、超參數 |
| 結果解讀 | 最後才使用測試集，使用圖表與指標檢查失敗樣本。 | [評估指標](../../名詞解釋/09_評估指標/README.md) | 過擬合與泛化 |

## 💡 為什麼採用「程式儲存格 ＋ AI 互動儲存格」？

外部 AI 從零生成程式時，常會遺漏資料切分、把測試集拿來調參，或因套件版本不同而無法執行。因此本範例先提供可執行的專業基底，再讓學生用 Prompt 深入追問。

1. **保證可執行的穩定基底**：筆記本已備妥資料載入、前處理、評估與推論流程。
2. **段段有解說**：每個程式儲存格下方都有白話說明，指出該步驟的目的與常見誤解。
3. **內建 AI 伴學對話小抄**：每個步驟提供觀念問題與安全調參方向；可直接複製給 AI。
4. **支援 Colab**：找不到 CSV 時，筆記本會自動從本專案下載資料。

## 📊 資料集介紹

| 項目 | 說明 |
| :--- | :--- |
| 資料檔 | [{dataset}](./{dataset})（[直接下載]({RAW}/{quote(folder)}/{quote(dataset)})） |
| 任務類型 | {task} |
| 目標 y | {y} |
| 核心演算法 | {algos} |
| 原始資料來源 | [{source.as_posix()}](../../{source.as_posix()}) |

## 🧭 機器學習全流程導覽（在 `{file}` 中逐步實作）

```text
[步驟 0：環境準備與繁體中文字型]
       │
       ▼
[步驟 1：資料載入與結構探索（EDA）]
       │
       ▼
[步驟 2：特徵關係、尺度與視覺化]
       │
       ▼
[步驟 3：前處理、Pipeline 與交叉驗證]
       │
       ▼
[步驟 4：測試集大考與指標解讀]
       │
       ▼
[步驟 5：模型診斷或客群輪廓]
       │
       ▼
[步驟 6：實務推論或決策模擬]
```

## 🚀 學習步驟

1. 開啟 [筆記本](./{file})，從步驟 0 依序執行。
2. 不要跳過 EDA、資料切分或交叉驗證。
3. 每段先自行思考，再複製下方 Prompt 向 AI 追問。
4. 只在訓練資料調參，測試集只在最後驗收。

## 🖥️ 進階專題：模型比較桌面 GUI

完成 Notebook 後，請執行 [app.py](./app.py)。它會載入同一份資料集，讓使用者選擇模型、抽取一筆資料並進行推論（分群範例則比較既有客戶在兩種分群法下的結果）。

### 📋 可複製給 AI 的延伸 Prompt

```text
請扮演機器學習教練。根據本範例的資料集、前處理與模型比較結果：
1. 用生活化比喻解釋各模型的差異與適用情境；
2. 只選一個超參數，設計不使用測試集的調參實驗；
3. 解釋 Accuracy／混淆矩陣或分群結果的限制；
4. 提出一個真實使用前必須補做的資料品質與風險檢查。
不要捏造資料，也不要宣稱結果可直接用於醫療、金融或其他高風險決策。
```

在本資料夾中執行：

```bash
python app.py
```

此 GUI 是教學與模型比較工具，不可作為真實領域的專業判斷。
""")

def app(spec):
 folder, file, title, dataset, source, target, task, algos, kind = spec
 if kind == "cluster":
  return dedent(f'''import tkinter as tk
from tkinter import ttk
from pathlib import Path
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).with_name("{dataset}")
df = pd.read_csv(DATA)
features = ["年齡", "收入(千)", "消費指數(1-100)"]
scaled = StandardScaler().fit_transform(df[features])
df["階層式群組"] = AgglomerativeClustering(n_clusters=4).fit_predict(scaled)
df["DBSCAN群組"] = DBSCAN(eps=.75, min_samples=5).fit_predict(scaled)

root = tk.Tk()
root.title("{title}｜AI 伴學工具")
root.geometry("820x540")
ttk.Label(root, text="{title}", font=("Arial", 18, "bold")).pack(pady=(18, 6))
ttk.Label(root, text="以既有顧客資料比較階層式集群與 DBSCAN；群組編號本身沒有高低意義。").pack()
output = tk.Text(root, width=94, height=20, wrap="word")
output.pack(padx=20, pady=14)

def show_case():
    row = df.sample(1).iloc[0]
    output.delete("1.0", tk.END)
    output.insert(tk.END, "【抽樣顧客】\\n" + row[features].to_string())
    output.insert(tk.END, f"\\n\\n階層式集群：群組 {{row['階層式群組']}}")
    output.insert(tk.END, f"\\nDBSCAN：群組 {{row['DBSCAN群組']}}（-1 代表雜訊）")
    output.insert(tk.END, "\\n\\n請回到 Notebook 對照群組平均輪廓，再提出可驗證的行銷假設。")

ttk.Button(root, text="抽取既有顧客並比較分群", command=show_case).pack()
show_case()
root.mainloop()
''')
 if kind == "classic":
  model_code = '''from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
models = {
    "KNN": Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(5))]),
    "決策樹": DecisionTreeClassifier(max_depth=4, random_state=42),
    "核化 SVM": Pipeline([("scale", StandardScaler()), ("model", SVC(C=2, kernel="rbf"))]),
    "高斯貝氏": GaussianNB(),
    "梯度提升": GradientBoostingClassifier(random_state=42),
    "Stacking": StackingClassifier(estimators=[
        ("knn", Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(5))])),
        ("tree", DecisionTreeClassifier(max_depth=4, random_state=42)), ("nb", GaussianNB())],
        final_estimator=LogisticRegression(max_iter=2000), cv=5),
}'''
 elif kind == "pca":
  model_code = '''from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
models = {
    "邏輯迴歸基準線": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000))]),
    "PCA + 核化 SVM": Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=.95, random_state=42)), ("model", SVC(C=1, kernel="rbf"))]),
}'''
 else:
  model_code = '''from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
models = {
    "邏輯迴歸基準線": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000))]),
    "MLP 神經網路": Pipeline([("scale", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(32,), early_stopping=True, validation_fraction=.2, n_iter_no_change=30, learning_rate_init=.005, max_iter=3000, random_state=42))]),
}'''
 return dedent(f"""import tkinter as tk
from tkinter import ttk
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).with_name("{dataset}")
df = pd.read_csv(DATA)
X = df.drop(columns="{target}")
y = df["{target}"]
{model_code}

root = tk.Tk()
root.title("{title}｜AI 伴學工具")
root.geometry("820x540")
ttk.Label(root, text="{title}", font=("Arial", 18, "bold")).pack(pady=(18, 6))
ttk.Label(root, text="選擇模型後，使用一筆抽樣資料進行即時推論；這是教學示範，不可取代專業判斷。").pack()

controls = ttk.Frame(root)
controls.pack(pady=12)
ttk.Label(controls, text="比較模型：").grid(row=0, column=0, padx=6)
selected = tk.StringVar(value=list(models)[0])
ttk.Combobox(controls, textvariable=selected, values=list(models), state="readonly", width=24).grid(row=0, column=1, padx=6)
output = tk.Text(root, width=94, height=20, wrap="word")
output.pack(padx=20, pady=8)

def predict_sample():
    row = X.sample(1)
    actual = y.loc[row.index[0]]
    name = selected.get()
    model = models[name].fit(X, y)
    prediction = model.predict(row)[0]
    output.delete("1.0", tk.END)
    output.insert(tk.END, "【模型】" + name + "\\n\\n【輸入特徵】\\n" + row.T.to_string(header=False))
    output.insert(tk.END, f"\\n\\n【模型預測】{{prediction}}")
    output.insert(tk.END, f"\\n【資料原始標籤（課後核對）】{{actual}}")
    output.insert(tk.END, "\\n\\n完整的交叉驗證、測試集評估與診斷圖，請在 Notebook 依序執行。")

ttk.Button(root, text="抽樣並執行模型預測", command=predict_sample).pack(pady=(0, 12))
predict_sample()
root.mainloop()
""")

for spec in SPECS:
 folder, file, title, dataset, source, *_ = spec
 path = ROOT / folder
 path.mkdir(parents=True, exist_ok=True)
 shutil.copy2(source, path/dataset)
 (path/"README.md").write_text(readme(spec), encoding="utf-8")
 (path/file).write_text(json.dumps(notebook(spec), ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
 (path/"app.py").write_text(app(spec), encoding="utf-8")
 print("created", path)
