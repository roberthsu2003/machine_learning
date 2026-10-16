#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人資管理 - 核心人才流失預警與員工關懷系統 (Tkinter 桌面 GUI 實戰版)
------------------------------------------------------------------
功能特點：
1. 完整機器學習全流程：自動載入員工滿意度與工時數據、特徵預處理 Pipeline、隨機森林模型訓練。
2. 內建實務人資情境：提供 3 組具代表性的員工留任情境（高工時倦怠、穩定核心主管、初入職場新人）一鍵載入。
3. 互動式介面：部門、考績下拉選單，與薪資、加班時數、滿意度滑桿即時連動。
4. 即時預測與人資關懷處遇：預測 6 個月內離職機率，並產出個別化留才與職涯面談指引。
"""

import os
import random
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score


def get_data_path():
    candidates = [
        os.path.join(os.path.dirname(__file__), "hr_turnover.csv"),
        os.path.join(os.getcwd(), "hr_turnover.csv"),
        os.path.join(os.getcwd(), "AI協作機器學習_職場實戰篇", "05_人資_員工離職風險預警", "hr_turnover.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class HRTurnoverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("👥 人資管理 - 員工離職風險預警與人才挽留系統")
        self.root.geometry("880x820")
        self.root.minsize(820, 760)

        self.pipeline = None
        self.metrics = {}

        self._build_ui()
        self._train_model()

    def _train_model(self):
        data_path = get_data_path()
        if not data_path:
            self.status_var.set("❌ 錯誤：找不到 hr_turnover.csv 資料集！請確認檔案位置。")
            messagebox.showerror("檔案遺失", "找不到 hr_turnover.csv 資料集，請確保其位於當前專案資料夾！")
            return

        try:
            df = pd.read_csv(data_path)
            X = df.drop(columns=["是否離職"])
            y = (df["是否離職"] == "是").astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            num_cols = ["年資_年", "月薪_TWD", "過去一年加班時數", "最近升遷距今年數", "工作滿意度評分"]
            cat_cols = ["所屬部門", "最近一次績效評等"]

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

            model_save_path = os.path.join(os.path.dirname(data_path), "model.joblib")
            try:
                joblib.dump(self.pipeline, model_save_path)
            except Exception:
                pass

            self.metrics = {"accuracy": acc, "auc": auc, "train_size": len(X_train), "test_size": len(X_test)}
            self.status_var.set(f"✅ 模型訓練就緒 ｜ 測試集準確率：{acc*100:.1f}% ｜ ROC-AUC：{auc:.3f} ｜ 樣本規模：{len(df):,} 筆")
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
            text="👥 企業人力資源處 — 員工離職風險預警與主動留才諮商系統",
            font=("Helvetica", 14, "bold"),
            foreground="#4338ca"
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="及早識別工作倦怠與轉職風險，協助 HR 與部門主管在遞出辭呈前啟動職涯面談與關懷",
            font=("Helvetica", 9),
            foreground="#475569"
        ).pack(anchor="w", pady=(2, 0))

        preset_frame = ttk.LabelFrame(main_frame, text="⚡ 快速人資評估情境 (點擊一鍵帶入)", padding="8")
        preset_frame.pack(fill=tk.X, pady=(0, 10))

        btn_box = ttk.Frame(preset_frame)
        btn_box.pack(fill=tk.X)

        ttk.Button(btn_box, text="🚨 情境 A：高工時倦怠警訊 (超量加班/滿意度低)", command=self._load_preset_high).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="⚖️ 情境 B：職涯停滯期 (升遷久未變動/績效優)", command=self._load_preset_medium).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🟢 情境 C：高度穩定核心幹部 (高滿意度/合理工時)", command=self._load_preset_low).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🎲 隨機抽樣員工", command=self._load_random_case).pack(side=tk.LEFT, padx=3)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        input_card = ttk.LabelFrame(content_frame, text="📋 員工在職狀態與考評特徵", padding="12")
        input_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.var_dept = tk.StringVar(value="研發工程")
        self.var_tenure = tk.DoubleVar(value=3.0)
        self.var_salary = tk.IntVar(value=65000)
        self.var_ot = tk.IntVar(value=320)
        self.var_perf = tk.StringVar(value="良好(B+)")
        self.var_promo = tk.DoubleVar(value=3.0)
        self.var_sat = tk.IntVar(value=2)

        r = 0
        ttk.Label(input_card, text="所屬部門：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_dept, values=["研發工程", "業務銷售", "行銷企劃", "客戶服務", "財務人資"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="公司年資 (年)：").grid(row=r, column=0, sticky="w", pady=4)
        f_ten = ttk.Frame(input_card)
        f_ten.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_ten, from_=0.5, to=12.0, variable=self.var_tenure, command=lambda v: self.lbl_ten.config(text=f"{float(v):.1f} 年")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_ten = ttk.Label(f_ten, text="3.0 年", width=8, anchor="e")
        self.lbl_ten.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="每月本薪 (TWD)：").grid(row=r, column=0, sticky="w", pady=4)
        f_sal = ttk.Frame(input_card)
        f_sal.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_sal, from_=30000, to=150000, variable=self.var_salary, command=lambda v: self.lbl_sal.config(text=f"${int(float(v)):,}")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_sal = ttk.Label(f_sal, text="$65,000", width=10, anchor="e")
        self.lbl_sal.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="過去一年加班時數：").grid(row=r, column=0, sticky="w", pady=4)
        f_ot = ttk.Frame(input_card)
        f_ot.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_ot, from_=20, to=450, variable=self.var_ot, command=lambda v: self.lbl_ot.config(text=f"{int(float(v))} 小時")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_ot = ttk.Label(f_ot, text="320 小時", width=10, anchor="e")
        self.lbl_ot.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="最近一次考績評等：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_perf, values=["特優(A)", "良好(B+)", "符合預期(B)", "需改善(C)"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="距上次升遷年數：").grid(row=r, column=0, sticky="w", pady=4)
        f_pro = ttk.Frame(input_card)
        f_pro.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_pro, from_=0.0, to=6.0, variable=self.var_promo, command=lambda v: self.lbl_pro.config(text=f"{float(v):.1f} 年")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_pro = ttk.Label(f_pro, text="3.0 年", width=8, anchor="e")
        self.lbl_pro.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="工作滿意度問卷 (1-5分)：").grid(row=r, column=0, sticky="w", pady=4)
        f_sat = ttk.Frame(input_card)
        f_sat.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_sat, from_=1, to=5, variable=self.var_sat, command=lambda v: self.lbl_sat.config(text=f"{int(float(v))} 分")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_sat = ttk.Label(f_sat, text="2 分", width=8, anchor="e")
        self.lbl_sat.pack(side=tk.RIGHT)

        r += 1
        btn_pred = tk.Button(
            input_card,
            text="🚀 啟動機器學習模型評估離職機率",
            bg="#4338ca",
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief="raised",
            cursor="hand2",
            pady=8,
            command=self._predict
        )
        btn_pred.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(15, 0))

        result_card = ttk.LabelFrame(content_frame, text="📊 離職風險分析與人資留才行動", padding="12")
        result_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        self.res_tag_var = tk.StringVar(value="尚未執行評估")
        self.res_tag_lbl = tk.Label(
            result_card,
            textvariable=self.res_tag_var,
            font=("Helvetica", 18, "bold"),
            fg="#64748b",
            bg="#f1f5f9",
            pady=12,
            relief="groove"
        )
        self.res_tag_lbl.pack(fill=tk.X, pady=(0, 10))

        self.prob_var = tk.StringVar(value="預估離職機率：-- %")
        self.prob_lbl = tk.Label(result_card, textvariable=self.prob_var, font=("Helvetica", 13, "bold"), fg="#1e293b")
        self.prob_lbl.pack(anchor="w", pady=(0, 4))

        self.prob_bar = ttk.Progressbar(result_card, orient="horizontal", mode="determinate", length=200)
        self.prob_bar.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(result_card, text="🤝 人資主管與部門主管關懷建議：", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.advice_txt = tk.Text(result_card, height=13, font=("Helvetica", 9), wrap="word", bg="#f8fafc", relief="solid", borderwidth=1)
        self.advice_txt.pack(fill=tk.BOTH, expand=True)
        self.advice_txt.insert("1.0", "請點選左側參數或上方情境，演算法將綜合考量滿意度、工時、薪資與升遷週期，產生專屬的人資訪談指引。")
        self.advice_txt.config(state="disabled")

        self.status_var = tk.StringVar(value="正在初始化機器學習環境與資料集...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w", padding="4")
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def _sync_labels(self):
        self.lbl_ten.config(text=f"{float(self.var_tenure.get()):.1f} 年")
        self.lbl_sal.config(text=f"${int(self.var_salary.get()):,}")
        self.lbl_ot.config(text=f"{int(self.var_ot.get())} 小時")
        self.lbl_pro.config(text=f"{float(self.var_promo.get()):.1f} 年")
        self.lbl_sat.config(text=f"{int(self.var_sat.get())} 分")

    def _load_preset_high(self):
        self.var_dept.set("客戶服務")
        self.var_tenure.set(2.0)
        self.var_salary.set(38000)
        self.var_ot.set(380)
        self.var_perf.set("符合預期(B)")
        self.var_promo.set(2.0)
        self.var_sat.set(1)
        self._sync_labels()
        self._predict()

    def _load_preset_medium(self):
        self.var_dept.set("研發工程")
        self.var_tenure.set(4.5)
        self.var_salary.set(78000)
        self.var_ot.set(220)
        self.var_perf.set("特優(A)")
        self.var_promo.set(4.0)
        self.var_sat.set(3)
        self._sync_labels()
        self._predict()

    def _load_preset_low(self):
        self.var_dept.set("業務銷售")
        self.var_tenure.set(5.0)
        self.var_salary.set(95000)
        self.var_ot.set(90)
        self.var_perf.set("特優(A)")
        self.var_promo.set(1.0)
        self.var_sat.set(5)
        self._sync_labels()
        self._predict()

    def _load_random_case(self):
        self.var_dept.set(random.choice(["研發工程", "業務銷售", "行銷企劃", "客戶服務", "財務人資"]))
        self.var_tenure.set(round(random.uniform(0.5, 8.0), 1))
        self.var_salary.set(random.randint(35000, 110000))
        self.var_ot.set(random.randint(50, 400))
        self.var_perf.set(random.choice(["特優(A)", "良好(B+)", "符合預期(B)", "需改善(C)"]))
        self.var_promo.set(round(random.uniform(0, 4), 1))
        self.var_sat.set(random.choice([1, 2, 3, 4, 5]))
        self._sync_labels()
        self._predict()

    def _predict(self):
        if not self.pipeline:
            messagebox.showwarning("模型未載入", "模型尚未訓練或載入完成！")
            return

        input_data = pd.DataFrame([{
            "所屬部門": self.var_dept.get(),
            "年資_年": float(self.var_tenure.get()),
            "月薪_TWD": int(self.var_salary.get()),
            "過去一年加班時數": int(self.var_ot.get()),
            "最近一次績效評等": self.var_perf.get(),
            "最近升遷距今年數": float(self.var_promo.get()),
            "工作滿意度評分": int(self.var_sat.get())
        }])

        try:
            prob = self.pipeline.predict_proba(input_data)[0][1]
            pct = prob * 100
            self.prob_bar["value"] = pct
            self.prob_var.set(f"預估半年內離職機率：{pct:.1f} %")

            if prob >= 0.65:
                self.res_tag_var.set("🚨 極高離職警戒 (需啟動一對一留才面談)")
                self.res_tag_lbl.config(fg="#b91c1c", bg="#fee2e2")
                advice = (
                    "【人資專案處遇建議：高風險挽留】\n"
                    "1. 立即啟動 HR 夥伴與部門直屬處長之一對一面談，傾聽滿意度低落核心原因。\n"
                    "2. 工時減壓機制：過去一年加班時數過高，建議檢討工作負載並啟動職務代理調配。\n"
                    "3. 薪酬競爭力檢驗：比對外部同業 P50/P75 薪資水準，評估專案留任獎金或調薪規劃。\n"
                    "4. 若為重要研發/主管人才，同步進行職務知識移轉防呆備援。"
                )
            elif prob >= 0.35:
                self.res_tag_var.set("⚖️ 中度流失傾向 (持續關注與職涯輔導)")
                self.res_tag_lbl.config(fg="#b45309", bg="#fef3c7")
                advice = (
                    "【人資專案處遇建議：溫和關懷】\n"
                    "1. 員工表現良好但可能面臨職涯成長停滯瓶頸。\n"
                    "2. 建議安排內部晉升審核或跨部門輪調專案，給予新舞台激發熱忱。\n"
                    "3. 主管應定期進行季度的 1 on 1 Check-in，建立心理安全感。"
                )
            else:
                self.res_tag_var.set("🟢 留任狀態良好 (安心發展核心骨幹)")
                self.res_tag_lbl.config(fg="#15803d", bg="#dcfce7")
                advice = (
                    "【人資專案處遇建議：高敬業度人才】\n"
                    "1. 目前身心與組織契合度極佳，是團隊的中流砥柱。\n"
                    "2. 可邀請擔任新人 Mentor（導師）或專案負責人，強化組織認同感。\n"
                    "3. 保持透明暢通的表揚獎勵機制，持續肯定其工作貢獻。"
                )

            self.advice_txt.config(state="normal")
            self.advice_txt.delete("1.0", tk.END)
            self.advice_txt.insert("1.0", advice)
            self.advice_txt.config(state="disabled")
        except Exception as e:
            messagebox.showerror("預測錯誤", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = HRTurnoverApp(root)
    root.mainloop()
