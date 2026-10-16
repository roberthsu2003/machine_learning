#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鐵達尼號生還預測系統 - 桌面 GUI 應用程式 (Tkinter 分類實戰版)
--------------------------------------------------------------
功能特點：
1. 完整二元分類 ML Pipeline：自動補值 (Imputation)、特徵編碼、標準化與邏輯迴歸訓練。
2. 內建實務預測情境：針對學生沒有測試資料的痛點，提供 3 組歷史代表情境與隨機乘客產生功能。
3. 互動式參數控制：艙等、性別、年齡滑桿與登船港口選擇。
4. 即時預測與歷史機率解析：輸出乘客生還機率百分比、分類結果與歷史決策洞察。
"""

import os
import sys
import random
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def get_data_path():
    """自動尋找 titanic.csv 檔案路徑"""
    candidates = [
        os.path.join(os.path.dirname(__file__), "titanic.csv"),
        os.path.join(os.getcwd(), "titanic.csv"),
        os.path.join(os.getcwd(), "AI提問與機器學習實戰", "02_鐵達尼號生存預測_分類篇", "titanic.csv"),
        os.path.join(os.path.dirname(__file__), "data", "titanic.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class TitanicPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚢 鐵達尼號乘客生還預測系統 - 二元分類實戰")
        self.root.geometry("820x760")
        self.root.minsize(760, 700)

        self.default_font = ("Helvetica", 11)
        self.title_font = ("Helvetica", 14, "bold")

        self.model_pipeline = None
        self.metrics = {}

        self._build_ui()
        self._train_model()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. 頂部標題與狀態
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        title_lbl = ttk.Label(
            header_frame,
            text="🚢 鐵達尼號乘客生還機率預測系統",
            font=("Helvetica", 16, "bold"),
            foreground="#1e3a8a",
        )
        title_lbl.pack(anchor=tk.W)

        subtitle_lbl = ttk.Label(
            header_frame,
            text="Scikit-Learn 二元分類 Pipeline ｜ 自動缺失值補齊 ｜ 模擬真實乘客特徵並計算生還機率",
            font=("Helvetica", 9),
            foreground="#64748b",
        )
        subtitle_lbl.pack(anchor=tk.W, pady=(2, 0))

        # 模型狀態面板
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
        preset_lf = ttk.LabelFrame(main_frame, text="💡 快速預測測試範例 (點擊一鍵套用歷史乘客情境)", padding=10)
        preset_lf.pack(fill=tk.X, pady=(0, 12))

        btn_box = ttk.Frame(preset_lf)
        btn_box.pack(fill=tk.X)

        btn_a = ttk.Button(
            btn_box,
            text="🌟 情境 A：頭等艙年輕名媛",
            command=lambda: self.load_preset("1st", "female", 26.0, "Cherbourg"),
        )
        btn_a.pack(side=tk.LEFT, padx=(0, 6), expand=True, fill=tk.X)

        btn_b = ttk.Button(
            btn_box,
            text="⚖️ 情境 B：二等艙中年紳士",
            command=lambda: self.load_preset("2nd", "male", 35.0, "Southampton"),
        )
        btn_b.pack(side=tk.LEFT, padx=6, expand=True, fill=tk.X)

        btn_c = ttk.Button(
            btn_box,
            text="📱 情境 C：三等艙移工少年",
            command=lambda: self.load_preset("3rd", "male", 20.0, "Queenstown"),
        )
        btn_c.pack(side=tk.LEFT, padx=6, expand=True, fill=tk.X)

        btn_random = ttk.Button(
            btn_box,
            text="🎲 隨機乘客",
            command=self.load_random_scenario,
        )
        btn_random.pack(side=tk.LEFT, padx=(6, 0), expand=True, fill=tk.X)

        # 3. 乘客特徵手動輸入區
        input_lf = ttk.LabelFrame(main_frame, text="⚙️ 乘客特徵參數手動設定", padding=12)
        input_lf.pack(fill=tk.X, pady=(0, 12))

        # 艙等
        ttk.Label(input_lf, text="艙等 (Pclass):", font=self.default_font).grid(
            row=0, column=0, sticky=tk.W, pady=6
        )
        self.pclass_var = tk.StringVar(value="1st")
        pclass_cb = ttk.Combobox(
            input_lf,
            textvariable=self.pclass_var,
            values=["1st", "2nd", "3rd"],
            state="readonly",
            width=20,
        )
        pclass_cb.grid(row=0, column=1, sticky=tk.W, padx=10, pady=6)

        # 生理性別
        ttk.Label(input_lf, text="生理性別 (Sex):", font=self.default_font).grid(
            row=1, column=0, sticky=tk.W, pady=6
        )
        self.sex_var = tk.StringVar(value="female")
        sex_cb = ttk.Combobox(
            input_lf,
            textvariable=self.sex_var,
            values=["female", "male"],
            state="readonly",
            width=20,
        )
        sex_cb.grid(row=1, column=1, sticky=tk.W, padx=10, pady=6)

        # 年齡 (滑桿 + 輸入框)
        self.age_var = tk.DoubleVar(value=28.0)
        self._create_slider_row(input_lf, "年齡 (Age):", self.age_var, 1.0, 75.0, 2)

        # 登船港口
        ttk.Label(input_lf, text="登船港口 (Embarked):", font=self.default_font).grid(
            row=3, column=0, sticky=tk.W, pady=6
        )
        self.embarked_var = tk.StringVar(value="Southampton")
        embarked_cb = ttk.Combobox(
            input_lf,
            textvariable=self.embarked_var,
            values=["Southampton", "Cherbourg", "Queenstown"],
            state="readonly",
            width=20,
        )
        embarked_cb.grid(row=3, column=1, sticky=tk.W, padx=10, pady=6)

        # 4. 操作按鈕
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=(4, 12))

        predict_btn = tk.Button(
            action_frame,
            text="🚀 計算生還機率與命運判定",
            font=("Helvetica", 12, "bold"),
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            padx=12,
            pady=8,
            relief=tk.FLAT,
            command=self.predict_survival,
        )
        predict_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        reset_btn = ttk.Button(
            action_frame,
            text="🔄 重設預設值",
            command=lambda: self.load_preset("1st", "female", 28.0, "Southampton"),
        )
        reset_btn.pack(side=tk.RIGHT, padx=(6, 0), ipady=6)

        # 5. 預測結果展示卡片
        result_lf = ttk.LabelFrame(main_frame, text="📊 模型預測結果與歷史機率解讀", padding=12)
        result_lf.pack(fill=tk.BOTH, expand=True)

        self.res_outcome_var = tk.StringVar(value="點擊上方按鈕開始預測")
        self.outcome_banner = ttk.Label(
            result_lf,
            textvariable=self.res_outcome_var,
            font=("Helvetica", 18, "bold"),
            foreground="#0f766e",
            anchor=tk.CENTER,
            background="#ccfbf1",
            padding=10,
        )
        self.outcome_banner.pack(fill=tk.X, pady=(0, 10))

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
            "1. 點擊上方的「快速預測測試範例」按鈕，快速載入不同社經背景之乘客數據。\n"
            "2. 您也可以透過滑桿自訂年齡、切換艙等與性別。\n"
            "3. 點擊「計算生還機率與命運判定」，系統將即時計算生還機率百分比與歷史洞察！"
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

    def load_preset(self, pclass, sex, age, embarked):
        self.pclass_var.set(pclass)
        self.sex_var.set(sex)
        self.age_var.set(age)
        self.embarked_var.set(embarked)
        self.predict_survival()

    def load_random_scenario(self):
        pclass = random.choice(["1st", "2nd", "3rd"])
        sex = random.choice(["female", "male"])
        age = round(random.uniform(5.0, 68.0), 0)
        embarked = random.choice(["Southampton", "Cherbourg", "Queenstown"])
        self.load_preset(pclass, sex, age, embarked)

    def _train_model(self):
        data_path = get_data_path()
        if not data_path:
            self.status_var.set("❌ 錯誤：找不到 titanic.csv 檔案！")
            messagebox.showerror("路徑錯誤", "找不到 titanic.csv 檔案！請確保檔案放置於專案目錄下。")
            return

        try:
            df = pd.read_csv(data_path)
            num_cols = ["age"]
            cat_cols = ["pclass", "sex", "embarked"]
            features = num_cols + cat_cols
            target_col = "survived"

            X = df[features]
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 數值管線：中位數補值 + 標準化
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            # 類別管線：眾數補值 + OneHotEncoder
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipe, num_cols),
                    ("cat", cat_pipe, cat_cols),
                ]
            )

            self.model_pipeline = Pipeline([
                ("prep", preprocessor),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ])

            self.model_pipeline.fit(X_train, y_train)

            # 評估測試集
            y_pred = self.model_pipeline.predict(X_test)
            y_prob = self.model_pipeline.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            self.status_var.set(
                f"✅ 模型訓練完成！樣本數: {len(df):,} 筆 ｜ 準確率 (Acc): {acc:.3f} ｜ 精確率: {prec:.3f} ｜ 召回率: {rec:.3f} ｜ ROC-AUC: {auc:.3f}"
            )

            self.predict_survival()

        except Exception as e:
            self.status_var.set(f"❌ 訓練失敗: {str(e)}")
            messagebox.showerror("訓練錯誤", f"模型訓練過程中發生錯誤：\n{str(e)}")

    def predict_survival(self):
        if self.model_pipeline is None:
            messagebox.showwarning("提示", "模型尚未訓練完成，請稍候！")
            return

        pclass = self.pclass_var.get()
        sex = self.sex_var.get()
        age = self.age_var.get()
        embarked = self.embarked_var.get()

        input_df = pd.DataFrame([{
            "pclass": pclass,
            "sex": sex,
            "age": age,
            "embarked": embarked,
        }])

        try:
            pred_class = self.model_pipeline.predict(input_df)[0]
            prob_survive = self.model_pipeline.predict_proba(input_df)[0][1] * 100.0

            if pred_class == 1:
                self.outcome_banner.config(foreground="#0f766e", background="#ccfbf1")
                self.res_outcome_var.set(f"✅ 判定生還！ 生還機率估算：{prob_survive:.1f}%")
            else:
                self.outcome_banner.config(foreground="#991b1b", background="#fee2e2")
                self.res_outcome_var.set(f"⚠️ 判定遇難！ 生還機率僅約：{prob_survive:.1f}%")

            self.res_detail_text.config(state=tk.NORMAL)
            self.res_detail_text.delete("1.0", tk.END)

            report = (
                f"【模擬乘客檔案】\n"
                f"  • 艙等：{pclass} 等艙\n"
                f"  • 性別：{'女性 (Female)' if sex == 'female' else '男性 (Male)'}\n"
                f"  • 年齡：{age:.0f} 歲\n"
                f"  • 登船港口：{embarked}\n"
                f"--------------------------------------------------\n"
                f"【歷史背景與機率洞察】\n"
                f"  • 預測生還機率：{prob_survive:.1f}% (臨界值門檻為 50.0%)\n"
                f"  • 核心影響因素：歷史上船長嚴格執行『婦孺優先 (Women and children first)』搜救準則，\n"
                f"    女性整體生還率高達 75% 以上，男性僅約 20%；此外，頭等艙乘客因位於上層甲板\n"
                f"    且具備優先撤離權限，生還率顯著高於底層三等艙乘客！"
            )

            self.res_detail_text.insert(tk.END, report)
            self.res_detail_text.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("預測錯誤", f"推論過程發生錯誤：\n{str(e)}")


def main():
    root = tk.Tk()
    app = TitanicPredictionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
