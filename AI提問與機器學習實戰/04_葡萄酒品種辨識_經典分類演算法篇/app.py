import tkinter as tk
from tkinter import ttk
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).with_name("wine.csv")
df = pd.read_csv(DATA)
X = df.drop(columns="酒種類")
y = df["酒種類"]
from sklearn.neighbors import KNeighborsClassifier
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
}

root = tk.Tk()
root.title("🍷 範例 04：葡萄酒品種辨識（經典分類演算法篇）｜AI 伴學工具")
root.geometry("820x540")
ttk.Label(root, text="🍷 範例 04：葡萄酒品種辨識（經典分類演算法篇）", font=("Arial", 18, "bold")).pack(pady=(18, 6))
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
    output.insert(tk.END, "【模型】" + name + "\n\n【輸入特徵】\n" + row.T.to_string(header=False))
    output.insert(tk.END, f"\n\n【模型預測】{prediction}")
    output.insert(tk.END, f"\n【資料原始標籤（課後核對）】{actual}")
    output.insert(tk.END, "\n\n完整的交叉驗證、測試集評估與診斷圖，請在 Notebook 依序執行。")

ttk.Button(root, text="抽樣並執行模型預測", command=predict_sample).pack(pady=(0, 12))
predict_sample()
root.mainloop()
