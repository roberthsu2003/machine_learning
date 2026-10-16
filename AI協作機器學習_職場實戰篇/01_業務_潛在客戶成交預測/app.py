#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
業務行銷 - B2B 潛在客戶成交機率預測系統 (Tkinter 桌面 GUI 實戰版)
------------------------------------------------------------------
功能特點：
1. 完整機器學習全流程：自動載入銷售管線資料、特徵預處理 Pipeline、模型訓練與即時推論。
2. 內建實務業務情境：提供 3 組具代表性的銷售商機一鍵載入與隨機情境產生，免去手動輸入。
3. 互動式介面：客戶產業、窗口職級、試用索取下拉選單，與預算、接洽次數、決策天數數值調整。
4. 即時預測與業務推進建議：預測成交機率，並產出個別化業務跟進策略與談判指引。
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
    """自動尋找 leads.csv 檔案路徑"""
    candidates = [
        os.path.join(os.path.dirname(__file__), "leads.csv"),
        os.path.join(os.getcwd(), "leads.csv"),
        os.path.join(os.getcwd(), "AI協作機器學習_職場實戰篇", "01_業務_潛在客戶成交預測", "leads.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class LeadConversionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("💼 B2B 業務銷售 - 潛在客戶成交機率預測系統")
        self.root.geometry("880x820")
        self.root.minsize(820, 760)

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
            self.status_var.set("❌ 錯誤：找不到 leads.csv 資料集！請確認檔案位置。")
            messagebox.showerror("檔案遺失", "找不到 leads.csv 資料集，請確保其位於當前專案資料夾！")
            return

        try:
            df = pd.read_csv(data_path)
            X = df.drop(columns=["是否成交"])
            y = (df["是否成交"] == "是").astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            num_cols = ["企業規模員工人數", "詢價預算_TWD", "歷史接洽次數", "預估決策天數"]
            cat_cols = ["客戶產業", "窗口職級", "是否索取產品試用", "是否參加線上說明會"]

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

            # 保存模型檔
            model_save_path = os.path.join(os.path.dirname(data_path), "model.joblib")
            try:
                joblib.dump(self.pipeline, model_save_path)
            except Exception:
                pass

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

        # 1. 頂部標題與業務識別
        header = ttk.Frame(main_frame)
        header.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            header,
            text="💼 B2B 業務銷售部門 — 潛在商機成交機率科學化量化評估系統",
            font=("Helvetica", 14, "bold"),
            foreground="#1e3a8a"
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="輸入商機特徵或選擇情境，AI 即時評估簽約成交機率並產出業務跟進處遇建議",
            font=("Helvetica", 9),
            foreground="#475569"
        ).pack(anchor="w", pady=(2, 0))

        # 2. 快捷範例按鈕列
        preset_frame = ttk.LabelFrame(main_frame, text="⚡ 快速預測範例情境 (點擊一鍵帶入)", padding="8")
        preset_frame.pack(fill=tk.X, pady=(0, 10))

        btn_box = ttk.Frame(preset_frame)
        btn_box.pack(fill=tk.X)

        ttk.Button(
            btn_box,
            text="🌟 情境 A：高成交機率個案 (大型金控/高層接洽)",
            command=self._load_preset_high
        ).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)

        ttk.Button(
            btn_box,
            text="⚖️ 情境 B：一般觀望個案 (中型製造/一般主管)",
            command=self._load_preset_medium
        ).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)

        ttk.Button(
            btn_box,
            text="⚠️ 情境 C：低機率高流失個案 (微型零售/無試用)",
            command=self._load_preset_low
        ).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)

        ttk.Button(
            btn_box,
            text="🎲 隨機產生個案",
            command=self._load_random_case
        ).pack(side=tk.LEFT, padx=3)

        # 3. 中間內容區 (左邊輸入表單，右邊預測結果)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # 左側：商機特徵輸入
        input_card = ttk.LabelFrame(content_frame, text="📝 商機與客戶特徵設定", padding="12")
        input_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        # 欄位變數
        self.var_industry = tk.StringVar(value="金融保險")
        self.var_employees = tk.IntVar(value=500)
        self.var_budget = tk.IntVar(value=450000)
        self.var_contacts = tk.IntVar(value=4)
        self.var_role = tk.StringVar(value="總經理/負責人")
        self.var_trial = tk.StringVar(value="是")
        self.var_webinar = tk.StringVar(value="是")
        self.var_decision_days = tk.IntVar(value=45)

        # 表單網格
        r = 0
        ttk.Label(input_card, text="客戶產業：").grid(row=r, column=0, sticky="w", pady=4)
        industry_cb = ttk.Combobox(
            input_card,
            textvariable=self.var_industry,
            values=["金融保險", "科技資訊", "醫療生技", "傳統製造", "批發零售", "專業服務"],
            state="readonly",
            width=18
        )
        industry_cb.grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="企業規模(員工人數)：").grid(row=r, column=0, sticky="w", pady=4)
        f_emp = ttk.Frame(input_card)
        f_emp.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_emp, from_=5, to=2000, variable=self.var_employees, command=lambda v: self.lbl_emp.config(text=f"{int(float(v))} 人")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_emp = ttk.Label(f_emp, text="500 人", width=8, anchor="e")
        self.lbl_emp.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="詢價預算 (TWD)：").grid(row=r, column=0, sticky="w", pady=4)
        f_bud = ttk.Frame(input_card)
        f_bud.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_bud, from_=20000, to=1000000, variable=self.var_budget, command=lambda v: self.lbl_bud.config(text=f"${int(float(v)):,}"))
        ttk.Scale(f_bud, from_=20000, to=1000000, variable=self.var_budget, command=lambda v: self.lbl_bud.config(text=f"${int(float(v)):,}")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_bud = ttk.Label(f_bud, text="$450,000", width=10, anchor="e")
        self.lbl_bud.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="歷史接洽次數：").grid(row=r, column=0, sticky="w", pady=4)
        f_cnt = ttk.Frame(input_card)
        f_cnt.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_cnt, from_=1, to=10, variable=self.var_contacts, command=lambda v: self.lbl_cnt.config(text=f"{int(float(v))} 次")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_cnt = ttk.Label(f_cnt, text="4 次", width=8, anchor="e")
        self.lbl_cnt.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="窗口決策職級：").grid(row=r, column=0, sticky="w", pady=4)
        role_cb = ttk.Combobox(
            input_card,
            textvariable=self.var_role,
            values=["總經理/負責人", "部門經理", "基層專員"],
            state="readonly",
            width=18
        )
        role_cb.grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="是否索取產品試用：").grid(row=r, column=0, sticky="w", pady=4)
        trial_cb = ttk.Combobox(
            input_card,
            textvariable=self.var_trial,
            values=["是", "否"],
            state="readonly",
            width=18
        )
        trial_cb.grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="是否參加線上說明會：").grid(row=r, column=0, sticky="w", pady=4)
        webinar_cb = ttk.Combobox(
            input_card,
            textvariable=self.var_webinar,
            values=["是", "否"],
            state="readonly",
            width=18
        )
        webinar_cb.grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="預估決策天數：").grid(row=r, column=0, sticky="w", pady=4)
        f_dec = ttk.Frame(input_card)
        f_dec.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_dec, from_=7, to=120, variable=self.var_decision_days, command=lambda v: self.lbl_dec.config(text=f"{int(float(v))} 天")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_dec = ttk.Label(f_dec, text="45 天", width=8, anchor="e")
        self.lbl_dec.pack(side=tk.RIGHT)

        # 預測觸發按鈕
        r += 1
        btn_pred = tk.Button(
            input_card,
            text="🚀 啟動機器學習模型評估成交機率",
            bg="#2563eb",
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief="raised",
            cursor="hand2",
            pady=8,
            command=self._predict
        )
        btn_pred.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(15, 0))

        # 右側：預測結果與處遇處置卡片
        result_card = ttk.LabelFrame(content_frame, text="📊 機器學習預測結果與處遇建議", padding="12")
        result_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        # 預測標籤大字
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

        # 機率百分比
        self.prob_var = tk.StringVar(value="預估成交機率：-- %")
        self.prob_lbl = tk.Label(
            result_card,
            textvariable=self.prob_var,
            font=("Helvetica", 13, "bold"),
            fg="#1e293b"
        )
        self.prob_lbl.pack(anchor="w", pady=(0, 4))

        # 進度條
        self.prob_bar = ttk.Progressbar(result_card, orient="horizontal", mode="determinate", length=200)
        self.prob_bar.pack(fill=tk.X, pady=(0, 12))

        # 處遇指引文字框
        ttk.Label(result_card, text="🎯 業務主管覆核與跟進策略建議：", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.advice_txt = tk.Text(result_card, height=13, font=("Helvetica", 9), wrap="word", bg="#f8fafc", relief="solid", borderwidth=1)
        self.advice_txt.pack(fill=tk.BOTH, expand=True)
        self.advice_txt.insert(
            "1.0",
            "點擊左側「啟動機器學習模型評估成交機率」或上方「情境按鈕」，系統將根據歷史 2,000 筆商機數據，產生精確的成交流向與主管行動方案建議。"
        )
        self.advice_txt.config(state="disabled")

        # 4. 底部狀態列
        self.status_var = tk.StringVar(value="正在初始化機器學習環境與資料集...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w", padding="4")
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def _sync_labels(self):
        self.lbl_emp.config(text=f"{int(self.var_employees.get())} 人")
        self.lbl_bud.config(text=f"${int(self.var_budget.get()):,}")
        self.lbl_cnt.config(text=f"{int(self.var_contacts.get())} 次")
        self.lbl_dec.config(text=f"{int(self.var_decision_days.get())} 天")

    def _load_preset_high(self):
        self.var_industry.set("金融保險")
        self.var_employees.set(1200)
        self.var_budget.set(650000)
        self.var_contacts.set(5)
        self.var_role.set("總經理/負責人")
        self.var_trial.set("是")
        self.var_webinar.set("是")
        self.var_decision_days.set(30)
        self._sync_labels()
        self._predict()

    def _load_preset_medium(self):
        self.var_industry.set("傳統製造")
        self.var_employees.set(150)
        self.var_budget.set(220000)
        self.var_contacts.set(3)
        self.var_role.set("部門經理")
        self.var_trial.set("是")
        self.var_webinar.set("否")
        self.var_decision_days.set(60)
        self._sync_labels()
        self._predict()

    def _load_preset_low(self):
        self.var_industry.set("批發零售")
        self.var_employees.set(15)
        self.var_budget.set(45000)
        self.var_contacts.set(1)
        self.var_role.set("基層專員")
        self.var_trial.set("否")
        self.var_webinar.set("否")
        self.var_decision_days.set(90)
        self._sync_labels()
        self._predict()

    def _load_random_case(self):
        self.var_industry.set(random.choice(["金融保險", "科技資訊", "醫療生技", "傳統製造", "批發零售", "專業服務"]))
        self.var_employees.set(random.randint(10, 1500))
        self.var_budget.set(random.randint(30000, 800000))
        self.var_contacts.set(random.randint(1, 8))
        self.var_role.set(random.choice(["總經理/負責人", "部門經理", "基層專員"]))
        self.var_trial.set(random.choice(["是", "否"]))
        self.var_webinar.set(random.choice(["是", "否"]))
        self.var_decision_days.set(random.randint(14, 100))
        self._sync_labels()
        self._predict()

    def _predict(self):
        if not self.pipeline:
            messagebox.showwarning("模型尚未載入", "模型尚未訓練或載入完成，請稍候再試。")
            return

        input_data = pd.DataFrame([{
            "客戶產業": self.var_industry.get(),
            "企業規模員工人數": int(self.var_employees.get()),
            "詢價預算_TWD": int(self.var_budget.get()),
            "歷史接洽次數": int(self.var_contacts.get()),
            "窗口職級": self.var_role.get(),
            "是否索取產品試用": self.var_trial.get(),
            "是否參加線上說明會": self.var_webinar.get(),
            "預估決策天數": int(self.var_decision_days.get())
        }])

        try:
            pred_class = self.pipeline.predict(input_data)[0]
            prob = self.pipeline.predict_proba(input_data)[0][1]
            pct = prob * 100

            self.prob_bar["value"] = pct
            self.prob_var.set(f"預估簽約成交機率：{pct:.1f} %")

            if prob >= 0.70:
                self.res_tag_var.set("🔥 高潛力成交商機 (建議主管跟進)")
                self.res_tag_lbl.config(fg="#15803d", bg="#dcfce7")
                advice = (
                    "【業務跟進處遇建議：高價值商機】\n"
                    "1. 立即指派資深業務主管陪同提案，鎖定痛點提出量身定做方案。\n"
                    "2. 由於窗口具決策影響力且預算充足，可在 3 天內提供具備時限誘因的專屬早鳥折扣合約。\n"
                    "3. 安排技術顧問進場進行 POC 概念驗證測試，加速內部簽呈流程。\n"
                    "4. 建議建立專屬 LINE/Teams 戰情溝通群，保持 24 小時快速答詢。"
                )
            elif prob >= 0.40:
                self.res_tag_var.set("⚖️ 中度潛在商機 (持續培養增強互信)")
                self.res_tag_lbl.config(fg="#b45309", bg="#fef3c7")
                advice = (
                    "【業務跟進處遇建議：溫熱型商機】\n"
                    "1. 客戶尚在多方比價與內部評估階段，切忌過度頻繁催促促成反效果。\n"
                    "2. 主動寄送同產業成功案例研討白皮書（特別強調 ROI 投資報酬率）。\n"
                    "3. 邀請該公司技術與營運團隊參加下週免費工作坊或產品體驗會。\n"
                    "4. 於接洽第 45 天設置追蹤節點，詢問內部評估遭遇之障礙並提供解答。"
                )
            else:
                self.res_tag_var.set("⚠️ 低成交傾向 (建議自動化行銷維持溫度)")
                self.res_tag_lbl.config(fg="#b91c1c", bg="#fee2e2")
                advice = (
                    "【業務跟進處遇建議：低急迫性商機】\n"
                    "1. 現階段投入過多一對一業務拜訪人力成本效益極低。\n"
                    "2. 將名單納入行銷自動化 EDM 電子報培育旅程，每月自動寄送產品更新資訊。\n"
                    "3. 觀察是否有重訪官網或重新下載試用之數位足跡，待活躍度提升再由業務重啟通話。\n"
                    "4. 檢視該客戶是否受限於預算，可評估提供精簡版月租方案或標準模板降低門檻。"
                )

            self.advice_txt.config(state="normal")
            self.advice_txt.delete("1.0", tk.END)
            self.advice_txt.insert("1.0", advice)
            self.advice_txt.config(state="disabled")

        except Exception as e:
            messagebox.showerror("預測異常", f"計算預測時發生錯誤：\n{str(e)}")


def main():
    root = tk.Tk()
    app = LeadConversionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
