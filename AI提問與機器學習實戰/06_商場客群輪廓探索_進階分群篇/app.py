import tkinter as tk
from tkinter import ttk
from pathlib import Path
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).with_name("customer.csv")
df = pd.read_csv(DATA)
features = ["年齡", "收入(千)", "消費指數(1-100)"]
scaled = StandardScaler().fit_transform(df[features])
df["階層式群組"] = AgglomerativeClustering(n_clusters=4).fit_predict(scaled)
df["DBSCAN群組"] = DBSCAN(eps=.75, min_samples=5).fit_predict(scaled)

root = tk.Tk()
root.title("🧭 範例 06：商場客群輪廓探索（進階分群篇）｜AI 伴學工具")
root.geometry("820x540")
ttk.Label(root, text="🧭 範例 06：商場客群輪廓探索（進階分群篇）", font=("Arial", 18, "bold")).pack(pady=(18, 6))
ttk.Label(root, text="以既有顧客資料比較階層式集群與 DBSCAN；群組編號本身沒有高低意義。").pack()
output = tk.Text(root, width=94, height=20, wrap="word")
output.pack(padx=20, pady=14)

def show_case():
    row = df.sample(1).iloc[0]
    output.delete("1.0", tk.END)
    output.insert(tk.END, "【抽樣顧客】\n" + row[features].to_string())
    output.insert(tk.END, f"\n\n階層式集群：群組 {row['階層式群組']}")
    output.insert(tk.END, f"\nDBSCAN：群組 {row['DBSCAN群組']}（-1 代表雜訊）")
    output.insert(tk.END, "\n\n請回到 Notebook 對照群組平均輪廓，再提出可驗證的行銷假設。")

ttk.Button(root, text="抽取既有顧客並比較分群", command=show_case).pack()
show_case()
root.mainloop()
