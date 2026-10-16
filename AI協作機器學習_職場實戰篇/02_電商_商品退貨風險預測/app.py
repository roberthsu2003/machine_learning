#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
電商營運 - 訂單退貨風險預警與逆物流控管系統 (Tkinter 桌面 GUI 實戰版)
------------------------------------------------------------------
功能特點：
1. 完整機器學習全流程：自動載入訂單數據、特徵預處理 Pipeline、隨機森林模型訓練與即時推論。
2. 內建實務電商情境：提供 3 組具代表性的訂單情境（高退貨服飾、低退貨美妝、常態購買）一鍵載入與隨機抽樣。
3. 互動式介面：商品類別、付款方式、配送方式下拉選單，與單價、折扣、退貨率滑桿雙向同步。
4. 即時預測與逆物流處遇回饋：預測退貨機率，並產出精準防退策略（出貨質檢、客服關懷、尺碼確認）。
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
        os.path.join(os.path.dirname(__file__), "returns.csv"),
        os.path.join(os.getcwd(), "returns.csv"),
        os.path.join(os.getcwd(), "AI協作機器學習_職場實戰篇", "02_電商_商品退貨風險預測", "returns.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class ReturnRiskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📦 電商營運 - 商品訂單退貨風險預測與防範系統")
        self.root.geometry("880x820")
        self.root.minsize(820, 760)

        self.pipeline = None
        self.metrics = {}

        self._build_ui()
        self._train_model()

    def _train_model(self):
        data_path = get_data_path()
        if not data_path:
            self.status_var.set("❌ 錯誤：找不到 returns.csv 資料集！請確認檔案位置。")
            messagebox.showerror("檔案遺失", "找不到 returns.csv 資料集，請確保其位於當前專案資料夾！")
            return

        try:
            df = pd.read_csv(data_path)
            X = df.drop(columns=["是否退貨"])
            y = (df["是否退貨"] == "是").astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            num_cols = ["單價_TWD", "折扣幅度", "過往退貨率", "購買數量"]
            cat_cols = ["商品類別", "是否首購顧客", "付款方式", "配送方式"]

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
            text="🛒 電商倉儲與營運團隊 — 訂單退貨風險科學化量化評估系統",
            font=("Helvetica", 14, "bold"),
            foreground="#0f766e"
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="出貨前評估訂單逆物流退貨風險，提早進行尺碼確認、出貨防呆與逆物流成本控管",
            font=("Helvetica", 9),
            foreground="#475569"
        ).pack(anchor="w", pady=(2, 0))

        preset_frame = ttk.LabelFrame(main_frame, text="⚡ 快速預測範例情境 (點擊一鍵帶入)", padding="8")
        preset_frame.pack(fill=tk.X, pady=(0, 10))

        btn_box = ttk.Frame(preset_frame)
        btn_box.pack(fill=tk.X)

        ttk.Button(btn_box, text="🚨 情境 A：極高退貨風險 (服飾大促/高退貨率歷史)", command=self._load_preset_high).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="⚖️ 情境 B：中度風險個案 (居家配件/首購客戶)", command=self._load_preset_medium).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🟢 情境 C：極低退貨風險 (美妝消耗品/信用卡付款)", command=self._load_preset_low).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🎲 隨機產生訂單", command=self._load_random_case).pack(side=tk.LEFT, padx=3)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        input_card = ttk.LabelFrame(content_frame, text="📦 訂單與顧客特徵設定", padding="12")
        input_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.var_category = tk.StringVar(value="流行服飾")
        self.var_price = tk.IntVar(value=1980)
        self.var_discount = tk.DoubleVar(value=0.3)
        self.var_first_buy = tk.StringVar(value="是")
        self.var_past_return = tk.DoubleVar(value=0.45)
        self.var_quantity = tk.IntVar(value=2)
        self.var_payment = tk.StringVar(value="超商取貨付款")
        self.var_shipping = tk.StringVar(value="黑貓宅急便")

        r = 0
        ttk.Label(input_card, text="商品類別：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_category, values=["流行服飾", "鞋包配件", "3C數位家電", "居家生活", "美妝保養"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="商品單價 (TWD)：").grid(row=r, column=0, sticky="w", pady=4)
        f_pri = ttk.Frame(input_card)
        f_pri.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_pri, from_=200, to=10000, variable=self.var_price, command=lambda v: self.lbl_pri.config(text=f"${int(float(v)):,}")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_pri = ttk.Label(f_pri, text="$1,980", width=10, anchor="e")
        self.lbl_pri.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="促銷折扣幅度：").grid(row=r, column=0, sticky="w", pady=4)
        f_disc = ttk.Frame(input_card)
        f_disc.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_disc, from_=0.0, to=0.6, variable=self.var_discount, command=lambda v: self.lbl_disc.config(text=f"{int(float(v)*100)}% 折扣")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_disc = ttk.Label(f_disc, text="30% 折扣", width=10, anchor="e")
        self.lbl_disc.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="是否為首購顧客：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_first_buy, values=["是", "否"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="過往退貨率歷史：").grid(row=r, column=0, sticky="w", pady=4)
        f_ret = ttk.Frame(input_card)
        f_ret.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_ret, from_=0.0, to=1.0, variable=self.var_past_return, command=lambda v: self.lbl_ret.config(text=f"{float(v)*100:.0f}%")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_ret = ttk.Label(f_ret, text="45%", width=8, anchor="e")
        self.lbl_ret.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="購買件數：").grid(row=r, column=0, sticky="w", pady=4)
        f_qty = ttk.Frame(input_card)
        f_qty.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_qty, from_=1, to=6, variable=self.var_quantity, command=lambda v: self.lbl_qty.config(text=f"{int(float(v))} 件")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_qty = ttk.Label(f_qty, text="2 件", width=8, anchor="e")
        self.lbl_qty.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="付款方式：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_payment, values=["信用卡線上刷卡", "超商取貨付款", "LINE Pay", "貨到付款"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="配送方式：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_shipping, values=["黑貓宅急便", "超商純取貨", "品牌自配送"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        btn_pred = tk.Button(
            input_card,
            text="🚀 啟動機器學習模型評估退貨風險",
            bg="#0f766e",
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief="raised",
            cursor="hand2",
            pady=8,
            command=self._predict
        )
        btn_pred.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(15, 0))

        result_card = ttk.LabelFrame(content_frame, text="📊 退貨風險預測與防範對策", padding="12")
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

        self.prob_var = tk.StringVar(value="預估退貨機率：-- %")
        self.prob_lbl = tk.Label(result_card, textvariable=self.prob_var, font=("Helvetica", 13, "bold"), fg="#1e293b")
        self.prob_lbl.pack(anchor="w", pady=(0, 4))

        self.prob_bar = ttk.Progressbar(result_card, orient="horizontal", mode="determinate", length=200)
        self.prob_bar.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(result_card, text="🛡️ 倉儲出貨檢驗與客服處置建議：", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.advice_txt = tk.Text(result_card, height=13, font=("Helvetica", 9), wrap="word", bg="#f8fafc", relief="solid", borderwidth=1)
        self.advice_txt.pack(fill=tk.BOTH, expand=True)
        self.advice_txt.insert("1.0", "請點擊左側「啟動機器學習模型評估退貨風險」或上方情境，系統將輸出退貨風險等級與專屬逆物流處置對策。")
        self.advice_txt.config(state="disabled")

        self.status_var = tk.StringVar(value="正在初始化機器學習環境與資料集...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w", padding="4")
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def _sync_labels(self):
        self.lbl_pri.config(text=f"${int(self.var_price.get()):,}")
        self.lbl_disc.config(text=f"{int(self.var_discount.get()*100)}% 折扣")
        self.lbl_ret.config(text=f"{self.var_past_return.get()*100:.0f}%")
        self.lbl_qty.config(text=f"{int(self.var_quantity.get())} 件")

    def _load_preset_high(self):
        self.var_category.set("流行服飾")
        self.var_price.set(2980)
        self.var_discount.set(0.4)
        self.var_first_buy.set("是")
        self.var_past_return.set(0.65)
        self.var_quantity.set(3)
        self.var_payment.set("超商取貨付款")
        self.var_shipping.set("黑貓宅急便")
        self._sync_labels()
        self._predict()

    def _load_preset_medium(self):
        self.var_category.set("居家生活")
        self.var_price.set(1280)
        self.var_discount.set(0.2)
        self.var_first_buy.set("是")
        self.var_past_return.set(0.25)
        self.var_quantity.set(1)
        self.var_payment.set("LINE Pay")
        self.var_shipping.set("超商純取貨")
        self._sync_labels()
        self._predict()

    def _load_preset_low(self):
        self.var_category.set("美妝保養")
        self.var_price.set(790)
        self.var_discount.set(0.1)
        self.var_first_buy.set("否")
        self.var_past_return.set(0.05)
        self.var_quantity.set(1)
        self.var_payment.set("信用卡線上刷卡")
        self.var_shipping.set("超商純取貨")
        self._sync_labels()
        self._predict()

    def _load_random_case(self):
        self.var_category.set(random.choice(["流行服飾", "鞋包配件", "3C數位家電", "居家生活", "美妝保養"]))
        self.var_price.set(random.choice([299, 490, 790, 1280, 1980, 2980, 4500]))
        self.var_discount.set(round(random.choice([0.0, 0.1, 0.2, 0.3, 0.4]), 2))
        self.var_first_buy.set(random.choice(["是", "否"]))
        self.var_past_return.set(round(random.uniform(0.0, 0.8), 2))
        self.var_quantity.set(random.randint(1, 4))
        self.var_payment.set(random.choice(["信用卡線上刷卡", "超商取貨付款", "LINE Pay", "貨到付款"]))
        self.var_shipping.set(random.choice(["黑貓宅急便", "超商純取貨", "品牌自配送"]))
        self._sync_labels()
        self._predict()

    def _predict(self):
        if not self.pipeline:
            messagebox.showwarning("模型未載入", "模型尚未訓練或載入完成！")
            return

        input_data = pd.DataFrame([{
            "商品類別": self.var_category.get(),
            "單價_TWD": int(self.var_price.get()),
            "折扣幅度": float(self.var_discount.get()),
            "是否首購顧客": self.var_first_buy.get(),
            "過往退貨率": float(self.var_past_return.get()),
            "購買數量": int(self.var_quantity.get()),
            "付款方式": self.var_payment.get(),
            "配送方式": self.var_shipping.get()
        }])

        try:
            prob = self.pipeline.predict_proba(input_data)[0][1]
            pct = prob * 100
            self.prob_bar["value"] = pct
            self.prob_var.set(f"預估退貨機率：{pct:.1f} %")

            if prob >= 0.65:
                self.res_tag_var.set("🚨 極高退貨風險 (需即時干預)")
                self.res_tag_lbl.config(fg="#b91c1c", bg="#fee2e2")
                advice = (
                    "【電商處置建議：高風險退貨警戒】\n"
                    "1. 尺碼/款式事前確認：出貨前自動觸發 LINE/簡訊，提醒確認身型尺寸表。\n"
                    "2. 包裹品質防呆：倉儲品檢人員於出貨時加強二次質檢並拍照存證。\n"
                    "3. 隨箱附贈換貨免運貼紙：主動引導『換貨代替退貨』，保住訂單營收。\n"
                    "4. 若為超商取貨付款且歷史取貨率低，派發溫馨取件提醒避免逾期退回。"
                )
            elif prob >= 0.35:
                self.res_tag_var.set("⚖️ 中度退貨風險 (標準質檢與提醒)")
                self.res_tag_lbl.config(fg="#b45309", bg="#fef3c7")
                advice = (
                    "【電商處置建議：常規防護】\n"
                    "1. 採標準出貨流程，隨箱附上清晰的退換貨說明與客服 QR Code。\n"
                    "2. 於商品出庫通知中附上穿搭示範或保養指南，提升顧客開箱滿意度。\n"
                    "3. 追蹤包裹妥投後第 3 天的滿意度問卷，提早掌握客訴問題。"
                )
            else:
                self.res_tag_var.set("🟢 低退貨風險 (優良訂單快速通關)")
                self.res_tag_lbl.config(fg="#15803d", bg="#dcfce7")
                advice = (
                    "【電商處置建議：高價值安心訂單】\n"
                    "1. 此訂單退貨機率極低，倉儲可直接排入快速出貨通關通道。\n"
                    "2. 客戶忠誠度與留存意願高，可隨箱附贈下次購買專屬優惠券（促成回購）。\n"
                    "3. 於七天鑑賞期滿後自動邀請顧客撰寫好評換購物金。"
                )

            self.advice_txt.config(state="normal")
            self.advice_txt.delete("1.0", tk.END)
            self.advice_txt.insert("1.0", advice)
            self.advice_txt.config(state="disabled")
        except Exception as e:
            messagebox.showerror("預測錯誤", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = ReturnRiskApp(root)
    root.mainloop()
