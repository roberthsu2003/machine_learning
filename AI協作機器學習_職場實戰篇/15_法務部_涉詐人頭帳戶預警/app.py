#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
法務部打詐專案 - 涉詐人頭帳戶與異常金流預警系統 (Tkinter 桌面 GUI 實戰版)
-------------------------------------------------------------------------
功能特點：
1. 完整機器學習全流程：自動載入資料、特徵預處理 Pipeline、模型訓練與即時推論。
2. 內建實務涉詐情境：提供 3 組具代表性的帳戶樣態一鍵載入與隨機情境產生。
3. 互動式介面：薪轉、非約定網銀下拉選單，與快進快出次數、暴增倍數滑桿雙向同步。
4. 檢警防詐聯防：預測帳戶淪為詐欺人頭戶機率，自動產出檢察官/警政/銀行端處置行動指引。
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
    """自動尋找 fraud_accounts.csv 檔案路徑"""
    candidates = [
        os.path.join(os.path.dirname(__file__), "fraud_accounts.csv"),
        os.path.join(os.getcwd(), "fraud_accounts.csv"),
        os.path.join(os.getcwd(), "AI協作機器學習_職場實戰篇", "15_法務部_涉詐人頭帳戶預警", "fraud_accounts.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class FraudAccountApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🛡️ 法務部打詐專案 - 涉詐人頭帳戶與異常金流預警系統")
        self.root.geometry("860x830")
        self.root.minsize(800, 750)

        self.pipeline = None
        self.metrics = {}

        self._build_ui()
        self._train_model()

    def _train_model(self):
        data_path = get_data_path()
        if not data_path:
            self.status_var.set("❌ 錯誤：找不到 fraud_accounts.csv 資料集！")
            messagebox.showerror("檔案遺失", "找不到 fraud_accounts.csv 資料集，請確保其位於當前資料夾！")
            return

        try:
            df = pd.read_csv(data_path)
            X = df.drop(columns=["是否為涉詐人頭帳戶"])
            y = (df["是否為涉詐人頭帳戶"] == "是").astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            numeric_features = ["開戶天數", "開戶人年齡", "近30天跨區與深夜ATM提款比率", "單日快進快出交易次數", "單月入帳金額暴增倍數"]
            categorical_features = ["是否有固定薪轉紀錄", "是否開通網銀非約定轉帳", "聯徵中心信用異常註記"]

            preprocessor = ColumnTransformer(transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
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

            self.metrics = {"accuracy": acc, "auc": auc}
            self.status_var.set(
                f"✅ 模型訓練完成 ｜ 測試集準確率：{acc*100:.1f}% ｜ ROC-AUC：{auc:.3f} ｜ 樣本規模：{len(df):,} 筆"
            )
        except Exception as e:
            self.status_var.set(f"❌ 模型訓練失敗：{str(e)}")
            messagebox.showerror("訓練錯誤", f"訓練過程發生異常：\n{str(e)}")

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(main_frame)
        header.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            header,
            text="🛡️ 法務部打詐專案 — 金融機構可疑帳戶涉詐洗錢「人頭戶風險預警」系統",
            font=("Helvetica", 14, "bold"),
            foreground="#1e3a8a"
        ).pack(anchor=tk.W)

        ttk.Label(
            header,
            text="檢警調防詐聯防 ｜ 輔助檢察官與法遵人員於被害人報案前即時啟動告誡與控管",
            font=("Helvetica", 9),
            foreground="#64748b"
        ).pack(anchor=tk.W, pady=(2, 0))

        self.status_var = tk.StringVar(value="⏳ 正在載入公務資料與訓練隨機森林模型...")
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            font=("Helvetica", 9),
            background="#f1f5f9",
            padding=6
        )
        status_bar.pack(fill=tk.X, pady=(0, 10))

        # 預設情境按鈕區
        preset_group = ttk.LabelFrame(main_frame, text="⚡ 快速測試預設個案情境", padding="10")
        preset_group.pack(fill=tk.X, pady=(0, 10))

        btn_row = ttk.Frame(preset_group)
        btn_row.pack(fill=tk.X)

        ttk.Button(btn_row, text="情境 A：賣簿人頭戶（快進快出暴增20倍）", command=self._load_scenario_a).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="情境 B：一般薪轉老戶（正常低風險）", command=self._load_scenario_b).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="情境 C：年輕學生涉詐（深夜跨區頻繁）", command=self._load_scenario_c).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="🎲 隨機產生帳戶", command=self._load_random_scenario).pack(side=tk.LEFT, padx=3)

        # 特徵輸入區
        input_group = ttk.LabelFrame(main_frame, text="📋 帳戶基本屬性與金流動態特徵", padding="10")
        input_group.pack(fill=tk.X, pady=(0, 10))

        grid = ttk.Frame(input_group)
        grid.pack(fill=tk.X)

        # 左欄
        left = ttk.Frame(grid)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        ttk.Label(left, text="是否有固定薪轉紀錄：", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(2, 0))
        self.var_salary = tk.StringVar(value="否")
        self.combo_salary = ttk.Combobox(left, textvariable=self.var_salary, values=["是", "否"], state="readonly")
        self.combo_salary.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(left, text="是否開通網銀非約定轉帳：", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(2, 0))
        self.var_transfer = tk.StringVar(value="是")
        self.combo_trans = ttk.Combobox(left, textvariable=self.var_transfer, values=["是", "否"], state="readonly")
        self.combo_trans.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(left, text="聯徵中心信用異常註記 (強停/退票)：", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(2, 0))
        self.var_credit = tk.StringVar(value="是")
        self.combo_credit = ttk.Combobox(left, textvariable=self.var_credit, values=["是", "否"], state="readonly")
        self.combo_credit.pack(fill=tk.X, pady=(0, 8))

        f_days = ttk.Frame(left)
        f_days.pack(fill=tk.X, pady=2)
        ttk.Label(f_days, text="開戶天數 (天)：", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.var_days = tk.IntVar(value=30)
        self.lbl_days = ttk.Label(f_days, text="30 天", font=("Helvetica", 10, "bold"), foreground="#2563eb")
        self.lbl_days.pack(side=tk.RIGHT)
        ttk.Scale(left, from_=5, to=1500, orient=tk.HORIZONTAL, variable=self.var_days, command=lambda v: self.lbl_days.config(text=f"{int(float(v))} 天")).pack(fill=tk.X, pady=(0, 2))

        # 右欄
        right = ttk.Frame(grid)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        f_fast = ttk.Frame(right)
        f_fast.pack(fill=tk.X, pady=2)
        ttk.Label(f_fast, text="單日快進快出交易次數 (次)：", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.var_fast = tk.IntVar(value=8)
        self.lbl_fast = ttk.Label(f_fast, text="8 次", font=("Helvetica", 10, "bold"), foreground="#2563eb")
        self.lbl_fast.pack(side=tk.RIGHT)
        ttk.Scale(right, from_=0, to=25, orient=tk.HORIZONTAL, variable=self.var_fast, command=lambda v: self.lbl_fast.config(text=f"{int(float(v))} 次")).pack(fill=tk.X, pady=(0, 6))

        f_surge = ttk.Frame(right)
        f_surge.pack(fill=tk.X, pady=2)
        ttk.Label(f_surge, text="單月入帳金額暴增倍數 (倍)：", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.var_surge = tk.DoubleVar(value=18.0)
        self.lbl_surge = ttk.Label(f_surge, text="18.0 倍", font=("Helvetica", 10, "bold"), foreground="#2563eb")
        self.lbl_surge.pack(side=tk.RIGHT)
        ttk.Scale(right, from_=1.0, to=40.0, orient=tk.HORIZONTAL, variable=self.var_surge, command=lambda v: self.lbl_surge.config(text=f"{float(v):.1f} 倍")).pack(fill=tk.X, pady=(0, 6))

        f_night = ttk.Frame(right)
        f_night.pack(fill=tk.X, pady=2)
        ttk.Label(f_night, text="跨區與深夜ATM提款比率 (%)：", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.var_night = tk.DoubleVar(value=0.75)
        self.lbl_night = ttk.Label(f_night, text="75.0 %", font=("Helvetica", 10, "bold"), foreground="#2563eb")
        self.lbl_night.pack(side=tk.RIGHT)
        ttk.Scale(right, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.var_night, command=lambda v: self.lbl_night.config(text=f"{float(v)*100:.1f} %")).pack(fill=tk.X, pady=(0, 6))

        f_age = ttk.Frame(right)
        f_age.pack(fill=tk.X, pady=2)
        ttk.Label(f_age, text="開戶人年齡 (歲)：", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.var_age = tk.IntVar(value=22)
        self.lbl_age = ttk.Label(f_age, text="22 歲", font=("Helvetica", 10, "bold"), foreground="#2563eb")
        self.lbl_age.pack(side=tk.RIGHT)
        ttk.Scale(right, from_=18, to=70, orient=tk.HORIZONTAL, variable=self.var_age, command=lambda v: self.lbl_age.config(text=f"{int(float(v))} 歲")).pack(fill=tk.X, pady=(0, 2))

        # 預測按鈕
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(2, 8))
        self.btn_predict = tk.Button(
            btn_frame,
            text="🛡️ 立即進行「涉詐人頭帳戶」風險預警研判",
            font=("Helvetica", 12, "bold"),
            bg="#1d4ed8",
            fg="white",
            padx=12,
            pady=6,
            command=self.predict_fraud
        )
        self.btn_predict.pack(fill=tk.X)

        # 評估面板
        res_group = ttk.LabelFrame(main_frame, text="📊 機器學習風險預警與司法聯防行動建議", padding="10")
        res_group.pack(fill=tk.BOTH, expand=True)

        self.lbl_result_badge = tk.Label(
            res_group,
            text="請輸入帳戶情資特徵並點擊上方評估按鈕",
            font=("Helvetica", 13, "bold"),
            bg="#f1f5f9",
            fg="#475569",
            padx=10,
            pady=6,
            relief=tk.GROOVE
        )
        self.lbl_result_badge.pack(fill=tk.X, pady=(0, 6))

        self.lbl_prob_detail = ttk.Label(res_group, text="", font=("Helvetica", 10), foreground="#1e293b")
        self.lbl_prob_detail.pack(anchor=tk.W, pady=(0, 4))

        ttk.Label(res_group, text="檢警調與金融防詐聯防措施指引：", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        self.txt_advice = tk.Text(res_group, height=5, font=("Helvetica", 10), wrap=tk.WORD, relief=tk.SOLID, borderwidth=1)
        self.txt_advice.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

    def _load_scenario_a(self):
        self.var_days.set(25)
        self.var_age.set(22)
        self.var_salary.set("否")
        self.var_transfer.set("是")
        self.var_credit.set("是")
        self.var_fast.set(12)
        self.var_surge.set(22.0)
        self.var_night.set(0.80)
        self._refresh()
        self.predict_fraud()

    def _load_scenario_b(self):
        self.var_days.set(1200)
        self.var_age.set(46)
        self.var_salary.set("是")
        self.var_transfer.set("否")
        self.var_credit.set("否")
        self.var_fast.set(0)
        self.var_surge.set(1.2)
        self.var_night.set(0.05)
        self._refresh()
        self.predict_fraud()

    def _load_scenario_c(self):
        self.var_days.set(45)
        self.var_age.set(19)
        self.var_salary.set("否")
        self.var_transfer.set("是")
        self.var_credit.set("否")
        self.var_fast.set(4)
        self.var_surge.set(8.5)
        self.var_night.set(0.70)
        self._refresh()
        self.predict_fraud()

    def _load_random_scenario(self):
        self.var_days.set(random.choice([20, 60, 200, 800]))
        self.var_age.set(random.randint(19, 58))
        self.var_salary.set(random.choice(["是", "否"]))
        self.var_transfer.set(random.choice(["是", "否"]))
        self.var_credit.set(random.choice(["是", "否"]))
        self.var_fast.set(random.choice([0, 1, 3, 7, 14]))
        self.var_surge.set(round(random.uniform(1.0, 25.0), 1))
        self.var_night.set(round(random.uniform(0.05, 0.85), 2))
        self._refresh()
        self.predict_fraud()

    def _refresh(self):
        self.lbl_days.config(text=f"{self.var_days.get()} 天")
        self.lbl_fast.config(text=f"{self.var_fast.get()} 次")
        self.lbl_surge.config(text=f"{self.var_surge.get():.1f} 倍")
        self.lbl_night.config(text=f"{self.var_night.get()*100:.1f} %")
        self.lbl_age.config(text=f"{self.var_age.get()} 歲")

    def predict_fraud(self):
        if self.pipeline is None:
            messagebox.showwarning("模型未就緒", "機器學習模型尚未訓練完成！")
            return

        sample = pd.DataFrame([{
            "開戶天數": int(self.var_days.get()),
            "開戶人年齡": int(self.var_age.get()),
            "是否有固定薪轉紀錄": self.var_salary.get(),
            "近30天跨區與深夜ATM提款比率": float(self.var_night.get()),
            "單日快進快出交易次數": int(self.var_fast.get()),
            "單月入帳金額暴增倍數": float(self.var_surge.get()),
            "是否開通網銀非約定轉帳": self.var_transfer.get(),
            "聯徵中心信用異常註記": self.var_credit.get()
        }])

        try:
            prob = self.pipeline.predict_proba(sample)[0, 1]
        except Exception as e:
            messagebox.showerror("預測錯誤", f"推論異常：\n{str(e)}")
            return

        if prob >= 0.70:
            self.lbl_result_badge.config(
                text=f"🚨 警報：極高涉詐人頭帳戶風險 (High Fraud Risk) ｜ 涉詐機率: {prob*100:.1f}%",
                bg="#fef2f2",
                fg="#b91c1c"
            )
        elif prob >= 0.40:
            self.lbl_result_badge.config(
                text=f"⚠️ 警報：可疑異常活動帳戶 (Suspicious Account) ｜ 涉詐機率: {prob*100:.1f}%",
                bg="#fffbeb",
                fg="#b45309"
            )
        else:
            self.lbl_result_badge.config(
                text=f"✅ 判定：一般正常活動帳戶 (Normal Account) ｜ 涉詐機率: {prob*100:.1f}%",
                bg="#f0fdf4",
                fg="#15803d"
            )

        self.lbl_prob_detail.config(
            text=f"模型研判涉詐人頭戶機率：{prob*100:.1f}% ｜ 正常合法使用機率：{(1-prob)*100:.1f}%"
        )

        adv = []
        if prob >= 0.70:
            adv.append("• 【即刻聯防控管】：本帳戶具高度車手收購洗錢特徵，強烈建議依洗防法立即通報該管金融機構啟動『圈存或暫停非臨櫃交易』。")
        if int(self.var_fast.get()) >= 5:
            adv.append(f"• 【快進快出異常】：單日密集快進快出 {self.var_fast.get()} 次，符合水房洗錢過水特徵，建議檢警調調閱該帳戶前置匯入來源 IP 與受款帳號名單。")
        if float(self.var_surge.get()) >= 10.0:
            adv.append(f"• 【異常金額暴增】：單月入帳金額暴增 {self.var_surge.get():.1f} 倍，且無實質商業往來背景，建議通知開戶人到案說明金流來源。")
        if self.var_salary.get() == "否" and int(self.var_age.get()) <= 24 and prob >= 0.50:
            adv.append("• 【洗防告誡機制】：名下無固定工作年輕族群，若查無實際被害人匯款，仍應啟動洗錢防制法第 15-2 條之『警察機關行政告誡』程序。")
        if prob < 0.40:
            adv.append("• 帳戶各項指標處於常態波動區間，未達聯防管制門檻，建請持續進行例行性洗防監控即可。")

        self.txt_advice.delete("1.0", tk.END)
        self.txt_advice.insert(tk.END, "\n".join(adv))


def main():
    root = tk.Tk()
    app = FraudAccountApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
