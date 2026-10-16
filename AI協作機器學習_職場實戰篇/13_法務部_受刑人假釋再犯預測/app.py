#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
法務部矯正署 - 受刑人假釋再犯風險評估系統 (Tkinter 桌面 GUI 實戰版)
------------------------------------------------------------------
功能特點：
1. 完整機器學習全流程：自動載入公務資料、特徵預處理 Pipeline、模型訓練與即時推論。
2. 內建實務假審情境：提供 3 組具代表性的假釋個案一鍵載入與隨機情境產生，免去手動輸入。
3. 互動式介面：罪名、累進處遇、戒癮狀況下拉選單，與前科次數、年齡滑桿雙向同步。
4. 即時預測與矯正處遇回饋：預測兩年內再犯機率，並產出個別化更生保護與觀護處遇建議。
"""

import os
import random
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score


def get_data_path():
    """自動尋找 recidivism.csv 檔案路徑"""
    candidates = [
        os.path.join(os.path.dirname(__file__), "recidivism.csv"),
        os.path.join(os.getcwd(), "recidivism.csv"),
        os.path.join(os.getcwd(), "AI協作機器學習_職場實戰篇", "13_法務部_受刑人假釋再犯預測", "recidivism.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class ParoleRecidivismApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚖️ 法務部矯正署 - 受刑人假釋再犯風險評估系統")
        self.root.geometry("860x820")
        self.root.minsize(800, 750)

        # 核心 ML 變數
        self.pipeline = None
        self.metrics = {}

        # 建立 UI
        self._build_ui()

        # 啟動時自動載入資料並完成模型訓練
        self._train_model()

    def _train_model(self):
        """讀取資料並建立 Pipeline 進行訓練"""
        data_path = get_data_path()
        if not data_path:
            self.status_var.set("❌ 錯誤：找不到 recidivism.csv 資料集！請確認檔案位置。")
            messagebox.showerror("檔案遺失", "找不到 recidivism.csv 資料集，請確保其位於當前專案資料夾！")
            return

        try:
            df = pd.read_csv(data_path)
            X = df.drop(columns=["兩年內是否再犯"])
            y = (df["兩年內是否再犯"] == "是").astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            num_cols = ["初次入監年齡", "入監前科次數", "在監違規扣分次數", "刑期已執行比例"]
            cat_cols = ["罪名類別", "累進處遇級別", "完成戒癮處遇", "具家庭支持與固定居所"]

            preprocessor = ColumnTransformer(transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
            ])

            self.pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
            ])

            self.pipeline.fit(X_train, y_train)

            y_pred = self.pipeline.predict(X_test)
            y_prob = self.pipeline.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            self.metrics = {
                "accuracy": acc,
                "auc": auc,
                "train_size": len(X_train),
                "test_size": len(X_test)
            }

            self.status_var.set(
                f"✅ 模型訓練就緒 ｜ 測試集準確率：{acc*100:.1f}% ｜ ROC-AUC：{auc:.3f} ｜ 樣本規模：{len(df):,} 筆"
            )
        except Exception as e:
            self.status_var.set(f"❌ 模型訓練失敗：{str(e)}")
            messagebox.showerror("訓練錯誤", f"訓練過程發生異常：\n{str(e)}")

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. 頂部標題與公務識別
        header = ttk.Frame(main_frame)
        header.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            header,
            text="🏛️ 法務部矯正署 假釋審查委員會 — 受刑人再犯風險科學化量化評估系統",
            font=("Helvetica", 14, "bold"),
            foreground="#1e3a8a"
        ).pack(anchor=tk.W)

        ttk.Label(
            header,
            text="公務資料分析實戰 ｜ 支援 Scikit-Learn 隨機森林分類器 ｜ 提供假釋委員會量化決策輔助",
            font=("Helvetica", 9),
            foreground="#64748b"
        ).pack(anchor=tk.W, pady=(2, 0))

        # 狀態列
        self.status_var = tk.StringVar(value="⏳ 正在載入公務資料與訓練隨機森林模型...")
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            font=("Helvetica", 9),
            background="#f1f5f9",
            padding=6
        )
        status_bar.pack(fill=tk.X, pady=(0, 10))

        # 2. 快速情境按鈕區 (Preset Scenarios)
        preset_group = ttk.LabelFrame(main_frame, text="⚡ 快速測試預設個案情境 (免手動逐項調整)", padding="10")
        preset_group.pack(fill=tk.X, pady=(0, 10))

        btn_row = ttk.Frame(preset_group)
        btn_row.pack(fill=tk.X)

        ttk.Button(btn_row, text="情境 A：毒品累犯（典型高風險）", command=self._load_scenario_a).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="情境 B：初犯優良在監（典型低風險）", command=self._load_scenario_b).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="情境 C：酒駕累犯無家支持", command=self._load_scenario_c).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="🎲 隨機產生個案", command=self._load_random_scenario).pack(side=tk.LEFT, padx=3)

        # 3. 核心特徵調整區 (雙欄佈局)
        input_group = ttk.LabelFrame(main_frame, text="📋 受刑人個案考核特徵輸入", padding="10")
        input_group.pack(fill=tk.X, pady=(0, 10))

        grid_frame = ttk.Frame(input_group)
        grid_frame.pack(fill=tk.X)

        # 左欄：類別條件
        left_col = ttk.Frame(grid_frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        # 罪名類別
        ttk.Label(left_col, text="主要罪名類別：", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(2, 0))
        self.var_crime = tk.StringVar(value="毒品危害")
        self.combo_crime = ttk.Combobox(
            left_col,
            textvariable=self.var_crime,
            values=["毒品危害", "詐欺洗錢", "公共危險酒駕", "竊盜侵占", "暴力傷害"],
            state="readonly"
        )
        self.combo_crime.pack(fill=tk.X, pady=(0, 8))

        # 累進處遇級別
        ttk.Label(left_col, text="累進處遇考核級別：", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(2, 0))
        self.var_level = tk.StringVar(value="二級(良好)")
        self.combo_level = ttk.Combobox(
            left_col,
            textvariable=self.var_level,
            values=["一級(優良)", "二級(良好)", "三級(普通)", "四級(初入)"],
            state="readonly"
        )
        self.combo_level.pack(fill=tk.X, pady=(0, 8))

        # 完成戒癮處遇
        ttk.Label(left_col, text="是否完成戒癮團體輔導處遇：", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(2, 0))
        self.var_rehab = tk.StringVar(value="是")
        self.combo_rehab = ttk.Combobox(
            left_col,
            textvariable=self.var_rehab,
            values=["是", "否", "無戒癮需求"],
            state="readonly"
        )
        self.combo_rehab.pack(fill=tk.X, pady=(0, 8))

        # 家庭支持與居所
        ttk.Label(left_col, text="具固定居所與親友支持系統：", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(2, 0))
        self.var_support = tk.StringVar(value="是")
        self.combo_support = ttk.Combobox(
            left_col,
            textvariable=self.var_support,
            values=["是", "否"],
            state="readonly"
        )
        self.combo_support.pack(fill=tk.X, pady=(0, 2))

        # 右欄：數值與滑桿
        right_col = ttk.Frame(grid_frame)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 入監前科次數
        f_priors = ttk.Frame(right_col)
        f_priors.pack(fill=tk.X, pady=2)
        ttk.Label(f_priors, text="前科累犯次數 (次)：", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.var_priors = tk.IntVar(value=1)
        self.lbl_priors = ttk.Label(f_priors, text="1 次", font=("Helvetica", 10, "bold"), foreground="#2563eb")
        self.lbl_priors.pack(side=tk.RIGHT)
        scale_priors = ttk.Scale(right_col, from_=0, to=6, orient=tk.HORIZONTAL, variable=self.var_priors, command=self._sync_priors)
        scale_priors.pack(fill=tk.X, pady=(0, 6))

        # 初次入監年齡
        f_age = ttk.Frame(right_col)
        f_age.pack(fill=tk.X, pady=2)
        ttk.Label(f_age, text="初次入監年齡 (歲)：", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.var_age = tk.IntVar(value=35)
        self.lbl_age = ttk.Label(f_age, text="35 歲", font=("Helvetica", 10, "bold"), foreground="#2563eb")
        self.lbl_age.pack(side=tk.RIGHT)
        scale_age = ttk.Scale(right_col, from_=18, to=65, orient=tk.HORIZONTAL, variable=self.var_age, command=self._sync_age)
        scale_age.pack(fill=tk.X, pady=(0, 6))

        # 違規扣分次數
        f_viol = ttk.Frame(right_col)
        f_viol.pack(fill=tk.X, pady=2)
        ttk.Label(f_viol, text="在監違規扣分次數 (次)：", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.var_violations = tk.IntVar(value=0)
        self.lbl_viol = ttk.Label(f_viol, text="0 次", font=("Helvetica", 10, "bold"), foreground="#2563eb")
        self.lbl_viol.pack(side=tk.RIGHT)
        scale_viol = ttk.Scale(right_col, from_=0, to=4, orient=tk.HORIZONTAL, variable=self.var_violations, command=self._sync_viol)
        scale_viol.pack(fill=tk.X, pady=(0, 6))

        # 刑期已執行比例
        f_ratio = ttk.Frame(right_col)
        f_ratio.pack(fill=tk.X, pady=2)
        ttk.Label(f_ratio, text="刑期已執行比例 (%)：", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.var_ratio = tk.DoubleVar(value=0.75)
        self.lbl_ratio = ttk.Label(f_ratio, text="75.0 %", font=("Helvetica", 10, "bold"), foreground="#2563eb")
        self.lbl_ratio.pack(side=tk.RIGHT)
        scale_ratio = ttk.Scale(right_col, from_=0.50, to=0.95, orient=tk.HORIZONTAL, variable=self.var_ratio, command=self._sync_ratio)
        scale_ratio.pack(fill=tk.X, pady=(0, 2))

        # 4. 預測執行大按鈕
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(2, 8))

        self.btn_predict = tk.Button(
            btn_frame,
            text="⚖️ 立即進行受刑人假釋再犯風險評估",
            font=("Helvetica", 12, "bold"),
            bg="#1d4ed8",
            fg="white",
            relief=tk.RAISED,
            padx=12,
            pady=6,
            command=self.predict_recidivism
        )
        self.btn_predict.pack(fill=tk.X)

        # 5. 評估報告顯示區
        res_group = ttk.LabelFrame(main_frame, text="📊 機器學習評估結果與保護管束建議", padding="10")
        res_group.pack(fill=tk.BOTH, expand=True)

        self.lbl_result_badge = tk.Label(
            res_group,
            text="請點擊上方按鈕進行風險評估",
            font=("Helvetica", 13, "bold"),
            bg="#f1f5f9",
            fg="#475569",
            padx=10,
            pady=6,
            relief=tk.GROOVE
        )
        self.lbl_result_badge.pack(fill=tk.X, pady=(0, 6))

        self.lbl_prob_detail = ttk.Label(
            res_group,
            text="",
            font=("Helvetica", 10),
            foreground="#1e293b"
        )
        self.lbl_prob_detail.pack(anchor=tk.W, pady=(0, 4))

        ttk.Label(res_group, text="更生保護處遇與假釋審查建議意見：", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        self.txt_advice = tk.Text(res_group, height=5, font=("Helvetica", 10), wrap=tk.WORD, relief=tk.SOLID, borderwidth=1)
        self.txt_advice.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

    def _sync_priors(self, val):
        self.lbl_priors.config(text=f"{int(float(val))} 次")

    def _sync_age(self, val):
        self.lbl_age.config(text=f"{int(float(val))} 歲")

    def _sync_viol(self, val):
        self.lbl_viol.config(text=f"{int(float(val))} 次")

    def _sync_ratio(self, val):
        self.lbl_ratio.config(text=f"{float(val)*100:.1f} %")

    def _load_scenario_a(self):
        """情境 A：毒品累犯（典型高風險）"""
        self.var_crime.set("毒品危害")
        self.var_level.set("三級(普通)")
        self.var_rehab.set("否")
        self.var_support.set("否")
        self.var_priors.set(3)
        self.var_age.set(26)
        self.var_violations.set(2)
        self.var_ratio.set(0.55)
        self._refresh_sync()
        self.predict_recidivism()

    def _load_scenario_b(self):
        """情境 B：初犯優良在監（典型低風險）"""
        self.var_crime.set("詐欺洗錢")
        self.var_level.set("一級(優良)")
        self.var_rehab.set("無戒癮需求")
        self.var_support.set("是")
        self.var_priors.set(0)
        self.var_age.set(45)
        self.var_violations.set(0)
        self.var_ratio.set(0.85)
        self._refresh_sync()
        self.predict_recidivism()

    def _load_scenario_c(self):
        """情境 C：酒駕累犯無家支持"""
        self.var_crime.set("公共危險酒駕")
        self.var_level.set("二級(良好)")
        self.var_rehab.set("否")
        self.var_support.set("否")
        self.var_priors.set(2)
        self.var_age.set(38)
        self.var_violations.set(1)
        self.var_ratio.set(0.68)
        self._refresh_sync()
        self.predict_recidivism()

    def _load_random_scenario(self):
        """隨機產生情境"""
        crimes = ["毒品危害", "詐欺洗錢", "公共危險酒駕", "竊盜侵占", "暴力傷害"]
        levels = ["一級(優良)", "二級(良好)", "三級(普通)", "四級(初入)"]
        rehabs = ["是", "否", "無戒癮需求"]
        supports = ["是", "否"]

        self.var_crime.set(random.choice(crimes))
        self.var_level.set(random.choice(levels))
        self.var_rehab.set(random.choice(rehabs))
        self.var_support.set(random.choice(supports))
        self.var_priors.set(random.randint(0, 5))
        self.var_age.set(random.randint(20, 60))
        self.var_violations.set(random.randint(0, 3))
        self.var_ratio.set(round(random.uniform(0.50, 0.90), 2))
        self._refresh_sync()
        self.predict_recidivism()

    def _refresh_sync(self):
        self._sync_priors(self.var_priors.get())
        self._sync_age(self.var_age.get())
        self._sync_viol(self.var_violations.get())
        self._sync_ratio(self.var_ratio.get())

    def predict_recidivism(self):
        if self.pipeline is None:
            messagebox.showwarning("模型未就緒", "機器學習模型尚未訓練完成！")
            return

        # 整理輸入樣本 DataFrame
        sample = pd.DataFrame([{
            "罪名類別": self.var_crime.get(),
            "初次入監年齡": int(self.var_age.get()),
            "入監前科次數": int(self.var_priors.get()),
            "累進處遇級別": self.var_level.get(),
            "在監違規扣分次數": int(self.var_violations.get()),
            "刑期已執行比例": float(self.var_ratio.get()),
            "完成戒癮處遇": self.var_rehab.get(),
            "具家庭支持與固定居所": self.var_support.get()
        }])

        try:
            prob = self.pipeline.predict_proba(sample)[0, 1]
            pred = self.pipeline.predict(sample)[0]
        except Exception as e:
            messagebox.showerror("預測錯誤", f"推論過程發生異常：\n{str(e)}")
            return

        # 呈現結果
        if prob >= 0.65:
            self.lbl_result_badge.config(
                text=f"🚨 評估結果：高再犯風險個案 (High Recidivism Risk) ｜ 預估再犯率: {prob*100:.1f}%",
                bg="#fef2f2",
                fg="#b91c1c"
            )
        elif prob >= 0.40:
            self.lbl_result_badge.config(
                text=f"⚠️ 評估結果：中度再犯風險個案 (Medium Recidivism Risk) ｜ 預估再犯率: {prob*100:.1f}%",
                bg="#fffbeb",
                fg="#b45309"
            )
        else:
            self.lbl_result_badge.config(
                text=f"✅ 評估結果：低再犯風險個案 (Low Recidivism Risk) ｜ 預估再犯率: {prob*100:.1f}%",
                bg="#f0fdf4",
                fg="#15803d"
            )

        self.lbl_prob_detail.config(
            text=f"預測統計機率值：再犯機率 {prob*100:.1f}% ｜ 未再犯機率 {(1-prob)*100:.1f}%"
        )

        # 產出處遇建議
        adv = []
        if self.var_priors.get() >= 2:
            adv.append(f"• 【累犯控管】：個案前科達 {self.var_priors.get()} 次，具犯罪行為慣性，假釋期間應提高定期向地檢署觀護人報到頻率。")
        if self.var_support.get() == "否":
            adv.append("• 【更生安置】：缺乏穩定居所與親友支持，強烈建議出獄前轉介臺灣更生保護會中途之家，提供食宿與就業媒合。")
        if self.var_rehab.get() == "否":
            adv.append("• 【強制處遇】：尚未完成戒癮輔導，若核准假釋，保護管束命令應附帶命其定期至指定醫療院所進行戒癮門診與採尿檢驗。")
        if self.var_violations.get() >= 2:
            adv.append(f"• 【在監違規】：服刑期間違規扣分 {self.var_violations.get()} 次，顯示衝動控制力較弱，建議假審會審慎考量其悔改悛悔程度。")
        if float(self.var_ratio.get()) >= 0.80 and self.var_level.get() == "一級(優良)":
            adv.append("• 【教化良好】：已服刑逾 80% 且處遇達一級，在監教化成果顯著，建議可考量准予假釋並納入常規保護管束。")

        if not adv:
            adv.append("• 個案整體指標平穩，無顯著負面違規或高危險因數，建議依標準公務行政程序進行假釋審查。")

        self.txt_advice.delete("1.0", tk.END)
        self.txt_advice.insert(tk.END, "\n".join(adv))


def main():
    root = tk.Tk()
    app = ParoleRecidivismApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
