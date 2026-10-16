#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
法務部行政執行署 - 公法欠費案件「徵起可能性」預測與智慧分流系統 (Tkinter 桌面 GUI 實戰版)
------------------------------------------------------------------------------------
功能特點：
1. 完整機器學習全流程：自動載入資料、特徵預處理 Pipeline、模型訓練與即時推論。
2. 內建實務執行情境：提供 3 組具代表性的公法欠費案件一鍵載入與隨機情境產生。
3. 互動式介面：案件類別、健保投保狀態下拉選單，與欠款金額、財產筆數、完納率滑桿雙向同步。
4. 案件智慧分流：預測成功徵起機率，並輸出書記官與執行官之具體執行行動建議（薪扣/查封/拘提）。
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
    """自動尋找 execution_cases.csv 檔案路徑"""
    candidates = [
        os.path.join(os.path.dirname(__file__), "execution_cases.csv"),
        os.path.join(os.getcwd(), "execution_cases.csv"),
        os.path.join(os.getcwd(), "AI協作機器學習_職場實戰篇", "14_法務部_公法欠費徵起預測", "execution_cases.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class ExecutionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🏛️ 法務部行政執行署 - 公法欠費案件徵起預測與智慧分流系統")
        self.root.geometry("860x830")
        self.root.minsize(800, 750)

        self.pipeline = None
        self.metrics = {}

        self._build_ui()
        self._train_model()

    def _train_model(self):
        data_path = get_data_path()
        if not data_path:
            self.status_var.set("❌ 錯誤：找不到 execution_cases.csv 資料集！")
            messagebox.showerror("檔案遺失", "找不到 execution_cases.csv 資料集，請確保其位於當前資料夾！")
            return

        try:
            df = pd.read_csv(data_path)
            X = df.drop(columns=["是否成功徵起"])
            y = (df["是否成功徵起"] == "是").astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            numeric_features = ["滯納欠款金額_TWD", "義務人年齡", "名下財產車輛筆數", "過往歷史完納率", "移送前催繳次數"]
            categorical_features = ["案件類別", "義務人性質", "健保投保狀態"]

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
            text="🏛️ 法務部行政執行署 — 移送公法欠款案件「徵起可能性」預測系統",
            font=("Helvetica", 14, "bold"),
            foreground="#1e3a8a"
        ).pack(anchor=tk.W)

        ttk.Label(
            header,
            text="公務案件智慧分流 ｜ 協助書記官與執行官快速擬定最適執行處分策略",
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

        ttk.Button(btn_row, text="情境 A：欠稅上班族（固定薪扣高機率）", command=self._load_scenario_a).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="情境 B：酒駕拒繳慣犯（無薪無財產）", command=self._load_scenario_b).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="情境 C：法人公司大額欠稅", command=self._load_scenario_c).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="🎲 隨機產生案件", command=self._load_random_scenario).pack(side=tk.LEFT, padx=3)

        # 特徵輸入區
        input_group = ttk.LabelFrame(main_frame, text="📋 移送案件與義務人屬性特徵", padding="10")
        input_group.pack(fill=tk.X, pady=(0, 10))

        grid = ttk.Frame(input_group)
        grid.pack(fill=tk.X)

        # 左欄
        left = ttk.Frame(grid)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        ttk.Label(left, text="案件移送類別：", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(2, 0))
        self.var_case_type = tk.StringVar(value="綜合所得稅")
        self.combo_type = ttk.Combobox(
            left,
            textvariable=self.var_case_type,
            values=["綜合所得稅", "地價房屋稅", "交通酒駕罰鍰", "健保勞保欠費", "工商環保罰鍰"],
            state="readonly"
        )
        self.combo_type.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(left, text="義務人法律主體性質：", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(2, 0))
        self.var_debtor_type = tk.StringVar(value="個人自然人")
        self.combo_debtor = ttk.Combobox(
            left,
            textvariable=self.var_debtor_type,
            values=["個人自然人", "營利事業法人"],
            state="readonly"
        )
        self.combo_debtor.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(left, text="健保投保單位狀態：", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(2, 0))
        self.var_insurance = tk.StringVar(value="固定受僱(有固定薪資)")
        self.combo_ins = ttk.Combobox(
            left,
            textvariable=self.var_insurance,
            values=["固定受僱(有固定薪資)", "職業工會(無固定薪資)", "區公所投保(無業或弱勢)"],
            state="readonly"
        )
        self.combo_ins.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(left, text="滯納欠款金額 (NT$)：", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(2, 0))
        self.var_amount = tk.StringVar(value="65000")
        ttk.Entry(left, textvariable=self.var_amount).pack(fill=tk.X, pady=(0, 2))

        # 右欄
        right = ttk.Frame(grid)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        f_assets = ttk.Frame(right)
        f_assets.pack(fill=tk.X, pady=2)
        ttk.Label(f_assets, text="名下查有財產/車輛筆數：", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.var_assets = tk.IntVar(value=1)
        self.lbl_assets = ttk.Label(f_assets, text="1 筆", font=("Helvetica", 10, "bold"), foreground="#2563eb")
        self.lbl_assets.pack(side=tk.RIGHT)
        ttk.Scale(right, from_=0, to=5, orient=tk.HORIZONTAL, variable=self.var_assets, command=lambda v: self.lbl_assets.config(text=f"{int(float(v))} 筆")).pack(fill=tk.X, pady=(0, 6))

        f_ratio = ttk.Frame(right)
        f_ratio.pack(fill=tk.X, pady=2)
        ttk.Label(f_ratio, text="過往歷史移送完納率 (%)：", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.var_history_ratio = tk.DoubleVar(value=0.80)
        self.lbl_history = ttk.Label(f_ratio, text="80.0 %", font=("Helvetica", 10, "bold"), foreground="#2563eb")
        self.lbl_history.pack(side=tk.RIGHT)
        ttk.Scale(right, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.var_history_ratio, command=lambda v: self.lbl_history.config(text=f"{float(v)*100:.1f} %")).pack(fill=tk.X, pady=(0, 6))

        f_dunning = ttk.Frame(right)
        f_dunning.pack(fill=tk.X, pady=2)
        ttk.Label(f_dunning, text="移送前催繳次數 (次)：", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.var_dunning = tk.IntVar(value=1)
        self.lbl_dunning = ttk.Label(f_dunning, text="1 次", font=("Helvetica", 10, "bold"), foreground="#2563eb")
        self.lbl_dunning.pack(side=tk.RIGHT)
        ttk.Scale(right, from_=1, to=5, orient=tk.HORIZONTAL, variable=self.var_dunning, command=lambda v: self.lbl_dunning.config(text=f"{int(float(v))} 次")).pack(fill=tk.X, pady=(0, 6))

        f_age = ttk.Frame(right)
        f_age.pack(fill=tk.X, pady=2)
        ttk.Label(f_age, text="義務人/負責人年齡 (歲)：", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        self.var_age = tk.IntVar(value=45)
        self.lbl_age = ttk.Label(f_age, text="45 歲", font=("Helvetica", 10, "bold"), foreground="#2563eb")
        self.lbl_age.pack(side=tk.RIGHT)
        ttk.Scale(right, from_=20, to=75, orient=tk.HORIZONTAL, variable=self.var_age, command=lambda v: self.lbl_age.config(text=f"{int(float(v))} 歲")).pack(fill=tk.X, pady=(0, 2))

        # 預測按鈕
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(2, 8))
        self.btn_predict = tk.Button(
            btn_frame,
            text="🏛️ 立即進行公法欠費「徵起可能性」預測分析",
            font=("Helvetica", 12, "bold"),
            bg="#1d4ed8",
            fg="white",
            padx=12,
            pady=6,
            command=self.predict_execution
        )
        self.btn_predict.pack(fill=tk.X)

        # 評估面板
        res_group = ttk.LabelFrame(main_frame, text="📊 徵起機率預測報告與行政執行行動方針", padding="10")
        res_group.pack(fill=tk.BOTH, expand=True)

        self.lbl_result_badge = tk.Label(
            res_group,
            text="請輸入案件特徵並點擊上方評估按鈕",
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

        ttk.Label(res_group, text="行政執行處分建議方針（分流措施）：", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        self.txt_advice = tk.Text(res_group, height=5, font=("Helvetica", 10), wrap=tk.WORD, relief=tk.SOLID, borderwidth=1)
        self.txt_advice.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

    def _load_scenario_a(self):
        self.var_case_type.set("綜合所得稅")
        self.var_debtor_type.set("個人自然人")
        self.var_insurance.set("固定受僱(有固定薪資)")
        self.var_amount.set("45000")
        self.var_assets.set(2)
        self.var_history_ratio.set(0.85)
        self.var_dunning.set(1)
        self.var_age.set(40)
        self._refresh()
        self.predict_execution()

    def _load_scenario_b(self):
        self.var_case_type.set("交通酒駕罰鍰")
        self.var_debtor_type.set("個人自然人")
        self.var_insurance.set("區公所投保(無業或弱勢)")
        self.var_amount.set("180000")
        self.var_assets.set(0)
        self.var_history_ratio.set(0.0)
        self.var_dunning.set(4)
        self.var_age.set(31)
        self._refresh()
        self.predict_execution()

    def _load_scenario_c(self):
        self.var_case_type.set("工商環保罰鍰")
        self.var_debtor_type.set("營利事業法人")
        self.var_insurance.set("固定受僱(有固定薪資)")
        self.var_amount.set("1200000")
        self.var_assets.set(3)
        self.var_history_ratio.set(0.3)
        self.var_dunning.set(3)
        self.var_age.set(55)
        self._refresh()
        self.predict_execution()

    def _load_random_scenario(self):
        types = ["綜合所得稅", "地價房屋稅", "交通酒駕罰鍰", "健保勞保欠費", "工商環保罰鍰"]
        ins = ["固定受僱(有固定薪資)", "職業工會(無固定薪資)", "區公所投保(無業或弱勢)"]
        self.var_case_type.set(random.choice(types))
        self.var_debtor_type.set(random.choice(["個人自然人", "營利事業法人"]))
        self.var_insurance.set(random.choice(ins))
        self.var_amount.set(str(random.randint(20000, 800000)))
        self.var_assets.set(random.randint(0, 4))
        self.var_history_ratio.set(round(random.choice([0.0, 0.3, 0.6, 0.9]), 2))
        self.var_dunning.set(random.randint(1, 5))
        self.var_age.set(random.randint(25, 65))
        self._refresh()
        self.predict_execution()

    def _refresh(self):
        self.lbl_assets.config(text=f"{self.var_assets.get()} 筆")
        self.lbl_history.config(text=f"{self.var_history_ratio.get()*100:.1f} %")
        self.lbl_dunning.config(text=f"{self.var_dunning.get()} 次")
        self.lbl_age.config(text=f"{self.var_age.get()} 歲")

    def predict_execution(self):
        if self.pipeline is None:
            messagebox.showwarning("模型未就緒", "機器學習模型尚未訓練完成！")
            return

        try:
            amt = float(self.var_amount.get().replace(",", "").strip())
        except ValueError:
            messagebox.showerror("輸入錯誤", "請輸入有效的欠款金額！")
            return

        sample = pd.DataFrame([{
            "案件類別": self.var_case_type.get(),
            "滯納欠款金額_TWD": amt,
            "義務人性質": self.var_debtor_type.get(),
            "義務人年齡": int(self.var_age.get()),
            "健保投保狀態": self.var_insurance.get(),
            "名下財產車輛筆數": int(self.var_assets.get()),
            "過往歷史完納率": float(self.var_history_ratio.get()),
            "移送前催繳次數": int(self.var_dunning.get())
        }])

        try:
            prob = self.pipeline.predict_proba(sample)[0, 1]
        except Exception as e:
            messagebox.showerror("預測錯誤", f"推論異常：\n{str(e)}")
            return

        if prob >= 0.70:
            self.lbl_result_badge.config(
                text=f"🎯 案件分流：高徵起成功率個案 (High Recovery) ｜ 預估徵起率: {prob*100:.1f}%",
                bg="#f0fdf4",
                fg="#15803d"
            )
        elif prob >= 0.40:
            self.lbl_result_badge.config(
                text=f"⚠️ 案件分流：中度徵起可能性個案 (Medium Recovery) ｜ 預估徵起率: {prob*100:.1f}%",
                bg="#fffbeb",
                fg="#b45309"
            )
        else:
            self.lbl_result_badge.config(
                text=f"🚨 案件分流：難以徵起/高脫產風險個案 (Low Recovery) ｜ 預估徵起率: {prob*100:.1f}%",
                bg="#fef2f2",
                fg="#b91c1c"
            )

        self.lbl_prob_detail.config(
            text=f"預估徵起成功率：{prob*100:.1f}% ｜ 執行難以徵起機率：{(1-prob)*100:.1f}%"
        )

        adv = []
        if self.var_insurance.get() == "固定受僱(有固定薪資)":
            adv.append("• 【快速薪扣】：義務人具有固定受僱投保單位，建議書記官第一時間批次核發『薪資所得扣押命令』，徵起成本最低、效益最高。")
        if self.var_assets.get() >= 1:
            adv.append(f"• 【動產/不動產】：名下查有 {self.var_assets.get()} 筆財產或車輛，若薪扣不足清償，可優先辦理不動產囑託查封或通報警方協尋拖吊車輛拍賣。")
        if self.var_insurance.get() == "區公所投保(無業或弱勢)" and self.var_assets.get() == 0:
            adv.append("• 【弱勢轉介或核發憑證】：義務人無財產且設籍公所，若查無金融帳戶存款，可評估核發『執行憑證』暫結，避免虛耗公務行政資源。")
        if amt >= 500000 and prob < 0.45:
            adv.append("• 【重大專案列管】：欠款金額逾 50 萬元且徵起率偏低，具隱匿財產脫產嫌疑，建議執行官傳喚到處報告財產狀況，必要時限制出境或聲請拘提。")
        if self.var_case_type.get() == "交通酒駕罰鍰" and prob < 0.50:
            adv.append("• 【強力專案執行】：酒駕罰鍰案件屬社會關注重大公益事件，縱使徵起率低，仍應發動現場實地查封，貫徹執法威嚇力。")

        if not adv:
            adv.append("• 依標準行政執行程序發函金融機構查詢存款並定期清查財產。")

        self.txt_advice.delete("1.0", tk.END)
        self.txt_advice.insert(tk.END, "\n".join(adv))


def main():
    root = tk.Tk()
    app = ExecutionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
