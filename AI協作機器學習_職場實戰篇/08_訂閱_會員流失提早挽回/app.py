#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SaaS 訂閱制營運 - 會員流失預警與續約挽留決策系統 (Tkinter 桌面 GUI 實戰版)
------------------------------------------------------------------
功能特點：
1. 完整機器學習全流程：自動載入會員登入與活躍度數據、特徵預處理 Pipeline、不平衡隨機森林模型訓練。
2. 內建實務訂閱情境：提供 3 組具代表性的會員情境（高流失瀕危會員、穩定長約用戶、常態活躍會員）一鍵載入。
3. 互動式介面：方案別、自動扣款下拉選單，與登入天數、使用時數、抱怨次數數值調整。
4. 即時預警與挽留處遇：預測次月退訂流失機率，並產出專屬客服慰留、專屬折扣或功能教學挽留指引。
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
        os.path.join(os.path.dirname(__file__), "churn.csv"),
        os.path.join(os.getcwd(), "churn.csv"),
        os.path.join(os.getcwd(), "AI協作機器學習_職場實戰篇", "08_訂閱_會員流失提早挽回", "churn.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class ChurnPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔔 訂閱制會員營運 - 會員流失提早挽回系統 (不平衡分類)")
        self.root.geometry("880x820")
        self.root.minsize(820, 760)

        self.pipeline = None
        self.metrics = {}

        self._build_ui()
        self._train_model()

    def _train_model(self):
        data_path = get_data_path()
        if not data_path:
            self.status_var.set("❌ 錯誤：找不到 churn.csv 資料集！請確認檔案位置。")
            messagebox.showerror("檔案遺失", "找不到 churn.csv 資料集，請確保其位於當前專案資料夾！")
            return

        try:
            df = pd.read_csv(data_path)
            X = df.drop(columns=["次月是否流失"])
            y = (df["次月是否流失"] == "是").astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            num_cols = ["已訂閱月數", "最近30天登入天數", "上月使用時長_小時", "最近客服抱怨次數", "社群互動次數"]
            cat_cols = ["訂閱方案", "是否綁定自動扣款"]

            preprocessor = ColumnTransformer(transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
            ])

            self.pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42))
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
            self.status_var.set(f"✅ 模型訓練就緒 ｜ 不平衡類別加權 ｜ 測試集 ROC-AUC：{auc:.3f} ｜ 樣本規模：{len(df):,} 位訂戶")
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
            text="📱 SaaS 軟體與內容訂閱服務 — 會員次月退訂流失預警與挽回系統",
            font=("Helvetica", 14, "bold"),
            foreground="#be123c"
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="及早識別即將退訂之沉睡或受挫會員，在信用卡扣款日前 14 天啟動主動關懷挽留",
            font=("Helvetica", 9),
            foreground="#475569"
        ).pack(anchor="w", pady=(2, 0))

        preset_frame = ttk.LabelFrame(main_frame, text="⚡ 快速會員活躍情境 (點擊一鍵帶入)", padding="8")
        preset_frame.pack(fill=tk.X, pady=(0, 10))

        btn_box = ttk.Frame(preset_frame)
        btn_box.pack(fill=tk.X)

        ttk.Button(btn_box, text="🚨 情境 A：極高流失瀕危會員 (未綁自動扣款/零登入/曾客訴)", command=self._load_preset_high).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="⚖️ 情境 B：功能使用低迷會員 (訂閱不久/登入少)", command=self._load_preset_medium).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🟢 情境 C：高度忠誠核心用戶 (長期訂閱/高頻使用)", command=self._load_preset_low).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🎲 隨機抽樣訂戶", command=self._load_random_case).pack(side=tk.LEFT, padx=3)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        input_card = ttk.LabelFrame(content_frame, text="📋 會員方案與使用特徵設定", padding="12")
        input_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.var_plan = tk.StringVar(value="個人標準版")
        self.var_tenure = tk.IntVar(value=3)
        self.var_login = tk.IntVar(value=2)
        self.var_hrs = tk.DoubleVar(value=1.5)
        self.var_comp = tk.IntVar(value=2)
        self.var_auto = tk.StringVar(value="否")
        self.var_post = tk.IntVar(value=0)

        r = 0
        ttk.Label(input_card, text="訂閱方案級別：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_plan, values=["個人標準版", "專業進階版", "企業尊榮版"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="已連續訂閱 (月)：").grid(row=r, column=0, sticky="w", pady=4)
        f_ten = ttk.Frame(input_card)
        f_ten.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_ten, from_=1, to=36, variable=self.var_tenure, command=lambda v: self.lbl_ten.config(text=f"{int(float(v))} 個月")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_ten = ttk.Label(f_ten, text="3 個月", width=8, anchor="e")
        self.lbl_ten.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="最近30天登入天數：").grid(row=r, column=0, sticky="w", pady=4)
        f_log = ttk.Frame(input_card)
        f_log.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_log, from_=0, to=30, variable=self.var_login, command=lambda v: self.lbl_log.config(text=f"{int(float(v))} 天")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_log = ttk.Label(f_log, text="2 天", width=8, anchor="e")
        self.lbl_log.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="上月使用時長 (小時)：").grid(row=r, column=0, sticky="w", pady=4)
        f_hrs = ttk.Frame(input_card)
        f_hrs.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_hrs, from_=0.1, to=80.0, variable=self.var_hrs, command=lambda v: self.lbl_hrs.config(text=f"{float(v):.1f} 小時")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_hrs = ttk.Label(f_hrs, text="1.5 小時", width=10, anchor="e")
        self.lbl_hrs.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="近期客服抱怨次數：").grid(row=r, column=0, sticky="w", pady=4)
        f_cmp = ttk.Frame(input_card)
        f_cmp.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_cmp, from_=0, to=4, variable=self.var_comp, command=lambda v: self.lbl_cmp.config(text=f"{int(float(v))} 次")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_cmp = ttk.Label(f_cmp, text="2 次", width=8, anchor="e")
        self.lbl_cmp.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="是否綁定自動扣款：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_auto, values=["是", "否"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="社群/論壇互動次數：").grid(row=r, column=0, sticky="w", pady=4)
        f_pos = ttk.Frame(input_card)
        f_pos.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_pos, from_=0, to=10, variable=self.var_post, command=lambda v: self.lbl_pos.config(text=f"{int(float(v))} 次")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_pos = ttk.Label(f_pos, text="0 次", width=8, anchor="e")
        self.lbl_pos.pack(side=tk.RIGHT)

        r += 1
        btn_pred = tk.Button(
            input_card,
            text="🚀 啟動不平衡模型評估流失機率",
            bg="#be123c",
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief="raised",
            cursor="hand2",
            pady=8,
            command=self._predict
        )
        btn_pred.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(15, 0))

        result_card = ttk.LabelFrame(content_frame, text="📊 流失預警分析與續約挽留對策", padding="12")
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

        self.prob_var = tk.StringVar(value="預估退訂流失機率：-- %")
        self.prob_lbl = tk.Label(result_card, textvariable=self.prob_var, font=("Helvetica", 13, "bold"), fg="#1e293b")
        self.prob_lbl.pack(anchor="w", pady=(0, 4))

        self.prob_bar = ttk.Progressbar(result_card, orient="horizontal", mode="determinate", length=200)
        self.prob_bar.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(result_card, text="🎯 客服營運與挽留促銷處遇：", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.advice_txt = tk.Text(result_card, height=13, font=("Helvetica", 9), wrap="word", bg="#f8fafc", relief="solid", borderwidth=1)
        self.advice_txt.pack(fill=tk.BOTH, expand=True)
        self.advice_txt.insert("1.0", "調整左側參數或點選情境，模型將針對不平衡訂閱數據，預警即將流失的用戶並產出專屬續約優惠與客服主動關懷策略。")
        self.advice_txt.config(state="disabled")

        self.status_var = tk.StringVar(value="正在初始化機器學習環境與資料集...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w", padding="4")
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def _sync_labels(self):
        self.lbl_ten.config(text=f"{int(self.var_tenure.get())} 個月")
        self.lbl_log.config(text=f"{int(self.var_login.get())} 天")
        self.lbl_hrs.config(text=f"{float(self.var_hrs.get()):.1f} 小時")
        self.lbl_cmp.config(text=f"{int(self.var_comp.get())} 次")
        self.lbl_pos.config(text=f"{int(self.var_post.get())} 次")

    def _load_preset_high(self):
        self.var_plan.set("個人標準版")
        self.var_tenure.set(2)
        self.var_login.set(1)
        self.var_hrs.set(0.5)
        self.var_comp.set(2)
        self.var_auto.set("否")
        self.var_post.set(0)
        self._sync_labels()
        self._predict()

    def _load_preset_medium(self):
        self.var_plan.set("專業進階版")
        self.var_tenure.set(6)
        self.var_login.set(8)
        self.var_hrs.set(6.0)
        self.var_comp.set(0)
        self.var_auto.set("是")
        self.var_post.set(1)
        self._sync_labels()
        self._predict()

    def _load_preset_low(self):
        self.var_plan.set("企業尊榮版")
        self.var_tenure.set(18)
        self.var_login.set(24)
        self.var_hrs.set(45.0)
        self.var_comp.set(0)
        self.var_auto.set("是")
        self.var_post.set(5)
        self._sync_labels()
        self._predict()

    def _load_random_case(self):
        self.var_plan.set(random.choice(["個人標準版", "專業進階版", "企業尊榮版"]))
        self.var_tenure.set(random.randint(1, 24))
        self.var_login.set(random.randint(0, 28))
        self.var_hrs.set(round(random.uniform(0.5, 50.0), 1))
        self.var_comp.set(random.choice([0, 0, 0, 1, 2]))
        self.var_auto.set(random.choice(["是", "否"]))
        self.var_post.set(random.randint(0, 5))
        self._sync_labels()
        self._predict()

    def _predict(self):
        if not self.pipeline:
            messagebox.showwarning("模型未載入", "模型尚未訓練或載入完成！")
            return

        input_data = pd.DataFrame([{
            "訂閱方案": self.var_plan.get(),
            "已訂閱月數": int(self.var_tenure.get()),
            "最近30天登入天數": int(self.var_login.get()),
            "上月使用時長_小時": float(self.var_hrs.get()),
            "最近客服抱怨次數": int(self.var_comp.get()),
            "是否綁定自動扣款": self.var_auto.get(),
            "社群互動次數": int(self.var_post.get())
        }])

        try:
            prob = self.pipeline.predict_proba(input_data)[0][1]
            pct = prob * 100
            self.prob_bar["value"] = pct
            self.prob_var.set(f"預估次月流失機率：{pct:.1f} %")

            if prob >= 0.60:
                self.res_tag_var.set("🚨 極高流失警訊 (需於3天內主動出擊)")
                self.res_tag_lbl.config(fg="#be123c", bg="#ffe4e6")
                advice = (
                    "【訂閱營運處遇建議：高危急救】\n"
                    "1. 專屬客服主動電訪：針對近期的客服抱怨主動致歉，並由資深技術專員排除痛點。\n"
                    "2. 續約折價誘因：於自動續約日前派發『專屬續約 7 折半年度優惠券』，降低流失率。\n"
                    "3. 重新引導教學：若為使用時長過低，發送 3 分鐘核心價值教學影片，重新喚醒產品價值感。"
                )
            elif prob >= 0.30:
                self.res_tag_var.set("⚖️ 中度流失傾向 (活躍度下滑/需內容推播)")
                self.res_tag_lbl.config(fg="#b45309", bg="#fef3c7")
                advice = (
                    "【訂閱營運處遇建議：保溫提醒】\n"
                    "1. 活躍喚醒：推送其方案最新推出的進階功能特報與熱門模板。\n"
                    "2. 滿意度微問卷：於登入時跳出 1 題快速問卷，偵測是否遭遇使用瓶頸。\n"
                    "3. 引導綁定信用卡自動續訂，並提供綁定贈送 30 天額外特權獎勵。"
                )
            else:
                self.res_tag_var.set("🟢 高黏著健康會員 (續約無虞/可推升級)")
                self.res_tag_lbl.config(fg="#15803d", bg="#dcfce7")
                advice = (
                    "【訂閱營運處遇建議：向上銷售 (Upselling)】\n"
                    "1. 會員黏著度極高，次月流失機率微乎其微。\n"
                    "2. 適合引導升級至『年繳方案』（提供 2 個月免費折扣，鎖定一年 ARR 營收）。\n"
                    "3. 邀請成為品牌代言人或推薦好友領回饋金 (Referral Program)。"
                )

            self.advice_txt.config(state="normal")
            self.advice_txt.delete("1.0", tk.END)
            self.advice_txt.insert("1.0", advice)
            self.advice_txt.config(state="disabled")
        except Exception as e:
            messagebox.showerror("預測錯誤", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = ChurnPredictionApp(root)
    root.mainloop()
