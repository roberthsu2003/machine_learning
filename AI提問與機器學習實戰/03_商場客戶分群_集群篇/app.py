#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商場客戶價值分群系統 - 桌面 GUI 應用程式 (Tkinter 集群實戰版)
--------------------------------------------------------------
功能特點：
1. 完整非監督式學習 ML Pipeline：特徵標準化 (StandardScaler) 與 K-Means (K=5) 擬合。
2. 內建實務客群預測範例：針對學生手邊沒有測試資料的痛點，提供 3 組實務顧客情境與隨機顧客生成按鈕。
3. 互動式參數控制：年齡、年收入與消費指數滑桿調整。
4. 即時分群判定與客群畫像推薦：輸出客戶分群標籤、特徵分析與針對性商業行銷策略。
"""

import os
import sys
import random
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score


def get_data_path():
    """自動尋找 customer.csv 檔案路徑"""
    candidates = [
        os.path.join(os.path.dirname(__file__), "customer.csv"),
        os.path.join(os.getcwd(), "customer.csv"),
        os.path.join(os.getcwd(), "AI提問與機器學習實戰", "03_商場客戶分群_集群篇", "customer.csv"),
        os.path.join(os.path.dirname(__file__), "data", "customer.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class CustomerClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🛍️ 商場客戶價值分群系統 - 非監督式學習實戰")
        self.root.geometry("820x760")
        self.root.minsize(760, 700)

        self.default_font = ("Helvetica", 11)
        self.title_font = ("Helvetica", 14, "bold")

        self.pipeline = None
        self.cluster_profiles = {}
        self.features = ["收入(千)", "消費指數(1-100)"]

        self._build_ui()
        self._train_model()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. 頂部標題與模型狀態
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        title_lbl = ttk.Label(
            header_frame,
            text="🛍️ 商場客戶價值畫像與分群系統",
            font=("Helvetica", 16, "bold"),
            foreground="#1e3a8a",
        )
        title_lbl.pack(anchor=tk.W)

        subtitle_lbl = ttk.Label(
            header_frame,
            text="使用 Scikit-Learn K-Means (K=5) 集群管線 ｜ 自動特徵標準化 ｜ 即時客群歸類與精準行銷策略",
            font=("Helvetica", 9),
            foreground="#64748b",
        )
        subtitle_lbl.pack(anchor=tk.W, pady=(2, 0))

        # 狀態面板
        self.status_var = tk.StringVar(value="⏳ 模型訓練中...")
        self.status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            font=("Helvetica", 9),
            background="#f1f5f9",
            padding=6,
        )
        self.status_bar.pack(fill=tk.X, pady=(0, 12))

        # 2. 快速載入預設預測資料專區 (解決學生無資料問題)
        preset_lf = ttk.LabelFrame(main_frame, text="💡 快速預測測試範例 (點擊一鍵套用客群情境)", padding=10)
        preset_lf.pack(fill=tk.X, pady=(0, 12))

        btn_box = ttk.Frame(preset_lf)
        btn_box.pack(fill=tk.X)

        btn_a = ttk.Button(
            btn_box,
            text="🌟 情境 A：高收高消費 VIP",
            command=lambda: self.load_preset(92.0, 85.0, 32),
        )
        btn_a.pack(side=tk.LEFT, padx=(0, 6), expand=True, fill=tk.X)

        btn_b = ttk.Button(
            btn_box,
            text="⚖️ 情境 B：年輕衝動潮流族",
            command=lambda: self.load_preset(25.0, 78.0, 22),
        )
        btn_b.pack(side=tk.LEFT, padx=6, expand=True, fill=tk.X)

        btn_c = ttk.Button(
            btn_box,
            text="📱 情境 C：高收入保守高資產",
            command=lambda: self.load_preset(105.0, 18.0, 48),
        )
        btn_c.pack(side=tk.LEFT, padx=6, expand=True, fill=tk.X)

        btn_random = ttk.Button(
            btn_box,
            text="🎲 隨機顧客",
            command=self.load_random_scenario,
        )
        btn_random.pack(side=tk.LEFT, padx=(6, 0), expand=True, fill=tk.X)

        # 3. 顧客特徵手動輸入區
        input_lf = ttk.LabelFrame(main_frame, text="⚙️ 顧客屬性參數手動配置", padding=12)
        input_lf.pack(fill=tk.X, pady=(0, 12))

        # 年收入 (15 ~ 140 千)
        self.income_var = tk.DoubleVar(value=60.0)
        self._create_slider_row(input_lf, "年收入 (千美元/千台幣):", self.income_var, 15.0, 140.0, 0)

        # 消費指數 (1 ~ 100)
        self.score_var = tk.DoubleVar(value=50.0)
        self._create_slider_row(input_lf, "商場消費活躍指數 (1-100):", self.score_var, 1.0, 100.0, 1)

        # 年齡 (18 ~ 70 歲)
        self.age_var = tk.DoubleVar(value=35.0)
        self._create_slider_row(input_lf, "年齡 (歲):", self.age_var, 18.0, 70.0, 2)

        # 4. 操作按鈕
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=(4, 12))

        predict_btn = tk.Button(
            action_frame,
            text="🚀 判定客群歸類與行銷方案",
            font=("Helvetica", 12, "bold"),
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            padx=12,
            pady=8,
            relief=tk.FLAT,
            command=self.predict_cluster,
        )
        predict_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        reset_btn = ttk.Button(
            action_frame,
            text="🔄 重設預設值",
            command=lambda: self.load_preset(60.0, 50.0, 35),
        )
        reset_btn.pack(side=tk.RIGHT, padx=(6, 0), ipady=6)

        # 5. 結果展示卡片
        result_lf = ttk.LabelFrame(main_frame, text="📊 客群畫像歸屬與精準行銷建議", padding=12)
        result_lf.pack(fill=tk.BOTH, expand=True)

        self.res_cluster_var = tk.StringVar(value="點擊上方按鈕開始分群判定")
        self.cluster_banner = ttk.Label(
            result_lf,
            textvariable=self.res_cluster_var,
            font=("Helvetica", 18, "bold"),
            foreground="#0f766e",
            anchor=tk.CENTER,
            background="#ccfbf1",
            padding=10,
        )
        self.cluster_banner.pack(fill=tk.X, pady=(0, 10))

        self.res_detail_text = tk.Text(
            result_lf,
            height=7,
            font=("Menlo", 10),
            bg="#ffffff",
            fg="#1e293b",
            relief=tk.GROOVE,
            padx=10,
            pady=8,
        )
        self.res_detail_text.pack(fill=tk.BOTH, expand=True)
        self.res_detail_text.insert(
            tk.END,
            "【使用指引】\n"
            "1. 點擊上方的「快速預測測試範例」按鈕，快速帶入不同消費水平之顧客。\n"
            "2. 您也可以透過滑桿自訂年收入與消費活躍度。\n"
            "3. 點擊「判定客群歸類與行銷方案」，系統將透過 K-Means 自動歸納客群畫像！"
        )
        self.res_detail_text.config(state=tk.DISABLED)

    def _create_slider_row(self, parent, label_text, var, min_v, max_v, row_idx):
        ttk.Label(parent, text=label_text, font=self.default_font).grid(
            row=row_idx, column=0, sticky=tk.W, pady=6
        )

        slider = ttk.Scale(
            parent,
            from_=min_v,
            to=max_v,
            variable=var,
            orient=tk.HORIZONTAL,
            length=300,
            command=lambda v: entry_var.set(f"{float(v):.0f}"),
        )
        slider.grid(row=row_idx, column=1, padx=10, pady=6, sticky=tk.W)

        entry_var = tk.StringVar(value=f"{var.get():.0f}")
        entry = ttk.Entry(parent, textvariable=entry_var, width=6)
        entry.grid(row=row_idx, column=2, padx=5, pady=6)

        def on_entry_change(*_):
            try:
                val = float(entry_var.get())
                if min_v <= val <= max_v:
                    var.set(val)
            except ValueError:
                pass

        entry_var.trace_add("write", on_entry_change)

    def load_preset(self, income, score, age):
        self.income_var.set(income)
        self.score_var.set(score)
        self.age_var.set(age)
        self.predict_cluster()

    def load_random_scenario(self):
        income = round(random.uniform(16.0, 135.0), 0)
        score = round(random.uniform(5.0, 96.0), 0)
        age = round(random.uniform(19.0, 68.0), 0)
        self.load_preset(income, score, age)

    def _train_model(self):
        data_path = get_data_path()
        if not data_path:
            self.status_var.set("❌ 錯誤：找不到 customer.csv 檔案！")
            messagebox.showerror("路徑錯誤", "找不到 customer.csv 檔案！請確保檔案放置於專案目錄下。")
            return

        try:
            df = pd.read_csv(data_path)
            X = df[self.features]

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # 建立 K-Means (手肘法最佳 K=5)
            self.kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42, n_init=10)
            df["群體"] = self.kmeans.fit_predict(X_scaled)

            # 計算輪廓係數
            sil_score = silhouette_score(X_scaled, self.kmeans.labels_)

            # 動態為各群命名畫像
            centroids_orig = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
            for c_id, (inc, sc) in enumerate(centroids_orig):
                if inc > 70 and sc > 60:
                    tag = "💎 頂級高奢 VIP 客群 (高收入・高消費)"
                    strategy = "維護專屬尊榮管家、頂級 VIP 預購會、高單價奢華客製服務。"
                elif inc > 70 and sc <= 60:
                    tag = "💼 高收精明理財型 (高收入・低消費)"
                    strategy = "強調商品品質與長期投資保值價值，避免廉價降價感，推播高質感實用選品。"
                elif inc <= 40 and sc > 60:
                    tag = "⚡ 年輕潮流衝動型 (低收入・高消費)"
                    strategy = "提供零卡分期、社群聯名爆款、IG 網紅開箱推薦，刺激立即購買慾望。"
                elif inc <= 40 and sc <= 60:
                    tag = "🌱 平價精打細算型 (低收入・低消費)"
                    strategy = "發放民生剛需折價券、買一送一滿額贈、抗通膨促銷組合。"
                else:
                    tag = "⚖️ 大眾中堅主流客群 (中收入・中消費)"
                    strategy = "會員集點回饋制度、週末節慶滿千送百、主打性價比與回購率。"

                self.cluster_profiles[c_id] = {
                    "tag": tag,
                    "avg_income": inc,
                    "avg_score": sc,
                    "strategy": strategy,
                }

            self.status_var.set(
                f"✅ 集群模型訓練完成！會員筆數: {len(df):,} 筆 ｜ 最佳群數 K=5 ｜ 輪廓係數: {sil_score:.3f} ｜ 慣性 (Inertia): {self.kmeans.inertia_:.1f}"
            )

            self.predict_cluster()

        except Exception as e:
            self.status_var.set(f"❌ 訓練失敗: {str(e)}")
            messagebox.showerror("訓練錯誤", f"模型訓練過程中發生錯誤：\n{str(e)}")

    def predict_cluster(self):
        if not hasattr(self, "kmeans"):
            messagebox.showwarning("提示", "模型尚未訓練完成，請稍候！")
            return

        inc = self.income_var.get()
        sc = self.score_var.get()
        age = self.age_var.get()

        input_df = pd.DataFrame([{"收入(千)": inc, "消費指數(1-100)": sc}])
        input_scaled = self.scaler.transform(input_df)
        cluster_id = int(self.kmeans.predict(input_scaled)[0])


        profile = self.cluster_profiles.get(cluster_id, {
            "tag": f"第 {cluster_id} 群客群",
            "strategy": "一般性關懷維護。",
            "avg_income": inc,
            "avg_score": sc,
        })

        self.res_cluster_var.set(f"分群歸類：{profile['tag']}")

        self.res_detail_text.config(state=tk.NORMAL)
        self.res_detail_text.delete("1.0", tk.END)

        report = (
            f"【顧客屬性數據】\n"
            f"  • 年收入：{inc:.0f} 千元 ｜ 消費活躍指數：{sc:.0f} 分 ｜ 年齡：{age:.0f} 歲\n"
            f"  • 所屬群集代號：Cluster #{cluster_id}\n"
            f"  • 該群集中心特徵：平均年收入約 {profile['avg_income']:.1f} 千元，平均消費指數約 {profile['avg_score']:.1f} 分\n"
            f"--------------------------------------------------\n"
            f"【精準行銷與 CRM 營運建議】\n"
            f"  👉 策略方向：{profile['strategy']}\n"
            f"  • 幾何距離觀察：K-Means 透過歐氏距離將顧客精準投射至 5 大客群中心，\n"
            f"    能有效協助百貨商場與零售企業將有限行銷預算配置在最具回報潛力的客群上！"
        )

        self.res_detail_text.insert(tk.END, report)
        self.res_detail_text.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = CustomerClusteringApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
