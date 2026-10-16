#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行銷企劃 - 跨通路廣告預算配置與營收效益預測系統 (Tkinter 桌面 GUI 實戰版)
------------------------------------------------------------------
功能特點：
1. 完整機器學習全流程：自動載入跨通路廣告投放數據、特徵預處理 Pipeline、隨機森林迴歸模型訓練。
2. 內建實務行銷情境：提供 3 組具代表性的行銷檔期情境（雙11大促、平日基本投放、春節檔期）一鍵載入。
3. 互動式介面：Meta、Google、KOL、LINE 預算滑桿調整，即時試算總預算與預估 ROI。
4. 即時預測與行銷決策建議：預估總營業額，並產出跨渠道資源配置優化方針。
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error


def get_data_path():
    candidates = [
        os.path.join(os.path.dirname(__file__), "ads_sales.csv"),
        os.path.join(os.getcwd(), "ads_sales.csv"),
        os.path.join(os.getcwd(), "AI協作機器學習_職場實戰篇", "03_行銷_廣告預算效益預測", "ads_sales.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class AdsRevenueApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📈 行銷企劃 - 跨通路廣告預算效益與營收預測系統")
        self.root.geometry("900x840")
        self.root.minsize(840, 780)

        self.pipeline = None
        self.metrics = {}

        self._build_ui()
        self._train_model()

    def _train_model(self):
        data_path = get_data_path()
        if not data_path:
            self.status_var.set("❌ 錯誤：找不到 ads_sales.csv 資料集！請確認檔案位置。")
            messagebox.showerror("檔案遺失", "找不到 ads_sales.csv 資料集，請確保其位於當前專案資料夾！")
            return

        try:
            df = pd.read_csv(data_path)
            X = df.drop(columns=["預估營收_TWD"])
            y = df["預估營收_TWD"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            num_cols = ["Meta臉書廣告_TWD", "Google搜尋廣告_TWD", "KOL網紅行銷_TWD", "LINE官方推播_TWD", "品牌搜尋熱度", "促銷折價券發放數"]
            cat_cols = ["季節檔期"]

            preprocessor = ColumnTransformer(transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
            ])

            self.pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42))
            ])

            self.pipeline.fit(X_train, y_train)

            y_pred = self.pipeline.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            model_save_path = os.path.join(os.path.dirname(data_path), "model.joblib")
            try:
                joblib.dump(self.pipeline, model_save_path)
            except Exception:
                pass

            self.metrics = {"r2": r2, "mae": mae, "train_size": len(X_train), "test_size": len(X_test)}
            self.status_var.set(f"✅ 模型訓練就緒 ｜ 測試集解釋力 R²：{r2:.3f} ｜ 平均絕對誤差 MAE：${mae:,.0f} ｜ 樣本規模：{len(df):,} 筆")
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
            text="📊 品牌行銷與廣告策略團隊 — 預算效益與營業額預估系統 (迴歸篇)",
            font=("Helvetica", 14, "bold"),
            foreground="#1e40af"
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="模擬不同跨通路行銷預算配比與促銷力度，機器學習即時估算檔期營收並推算 ROAS 投資回報率",
            font=("Helvetica", 9),
            foreground="#475569"
        ).pack(anchor="w", pady=(2, 0))

        preset_frame = ttk.LabelFrame(main_frame, text="⚡ 快速行銷預算情境 (點擊一鍵帶入)", padding="8")
        preset_frame.pack(fill=tk.X, pady=(0, 10))

        btn_box = ttk.Frame(preset_frame)
        btn_box.pack(fill=tk.X)

        ttk.Button(btn_box, text="🔥 情境 A：雙11年度大促攻堅 (高額投放+重磅KOL)", command=self._load_preset_high).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🌿 情境 B：年中慶重點檔期 (平衡配比+LINE推播)", command=self._load_preset_medium).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="☕ 情境 C：平日維穩基本投放 (精準搜尋+保底預算)", command=self._load_preset_low).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🎲 隨機配置預算", command=self._load_random_case).pack(side=tk.LEFT, padx=3)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        input_card = ttk.LabelFrame(content_frame, text="💰 各通路行銷預算與檔期設定", padding="12")
        input_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.var_meta = tk.IntVar(value=80000)
        self.var_google = tk.IntVar(value=100000)
        self.var_kol = tk.IntVar(value=50000)
        self.var_line = tk.IntVar(value=15000)
        self.var_season = tk.StringVar(value="年中慶")
        self.var_trend = tk.IntVar(value=65)
        self.var_coupon = tk.IntVar(value=3000)

        r = 0
        ttk.Label(input_card, text="季節行銷檔期：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_season, values=["雙11大促", "年中慶", "春節連假", "平日檔期"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="Meta 臉書廣告 (TWD)：").grid(row=r, column=0, sticky="w", pady=4)
        f_meta = ttk.Frame(input_card)
        f_meta.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_meta, from_=5000, to=200000, variable=self.var_meta, command=lambda v: self.lbl_meta.config(text=f"${int(float(v)):,}")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_meta = ttk.Label(f_meta, text="$80,000", width=10, anchor="e")
        self.lbl_meta.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="Google 搜尋廣告 (TWD)：").grid(row=r, column=0, sticky="w", pady=4)
        f_goog = ttk.Frame(input_card)
        f_goog.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_goog, from_=5000, to=200000, variable=self.var_google, command=lambda v: self.lbl_goog.config(text=f"${int(float(v)):,}")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_goog = ttk.Label(f_goog, text="$100,000", width=10, anchor="e")
        self.lbl_goog.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="KOL 網紅合作 (TWD)：").grid(row=r, column=0, sticky="w", pady=4)
        f_kol = ttk.Frame(input_card)
        f_kol.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_kol, from_=0, to=250000, variable=self.var_kol, command=lambda v: self.lbl_kol.config(text=f"${int(float(v)):,}")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_kol = ttk.Label(f_kol, text="$50,000", width=10, anchor="e")
        self.lbl_kol.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="LINE 官方推播 (TWD)：").grid(row=r, column=0, sticky="w", pady=4)
        f_line = ttk.Frame(input_card)
        f_line.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_line, from_=1000, to=40000, variable=self.var_line, command=lambda v: self.lbl_line.config(text=f"${int(float(v)):,}")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_line = ttk.Label(f_line, text="$15,000", width=10, anchor="e")
        self.lbl_line.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="品牌自然搜尋熱度：").grid(row=r, column=0, sticky="w", pady=4)
        f_trend = ttk.Frame(input_card)
        f_trend.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_trend, from_=10, to=100, variable=self.var_trend, command=lambda v: self.lbl_trend.config(text=f"{int(float(v))} 分")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_trend = ttk.Label(f_trend, text="65 分", width=8, anchor="e")
        self.lbl_trend.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="促銷折價券發放數：").grid(row=r, column=0, sticky="w", pady=4)
        f_coup = ttk.Frame(input_card)
        f_coup.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_coup, from_=500, to=20000, variable=self.var_coupon, command=lambda v: self.lbl_coup.config(text=f"{int(float(v)):,} 張")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_coup = ttk.Label(f_coup, text="3,000 張", width=10, anchor="e")
        self.lbl_coup.pack(side=tk.RIGHT)

        r += 1
        btn_pred = tk.Button(
            input_card,
            text="🚀 啟動迴歸模型預估預期總營收",
            bg="#1d4ed8",
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief="raised",
            cursor="hand2",
            pady=8,
            command=self._predict
        )
        btn_pred.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(15, 0))

        result_card = ttk.LabelFrame(content_frame, text="📊 機器學習預估營收與 ROI 效益", padding="12")
        result_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        self.res_tag_var = tk.StringVar(value="尚未執行預估")
        self.res_tag_lbl = tk.Label(
            result_card,
            textvariable=self.res_tag_var,
            font=("Helvetica", 18, "bold"),
            fg="#1e40af",
            bg="#eff6ff",
            pady=12,
            relief="groove"
        )
        self.res_tag_lbl.pack(fill=tk.X, pady=(0, 10))

        self.roas_var = tk.StringVar(value="預估廣告投資報酬率 ROAS：-- 倍")
        self.roas_lbl = tk.Label(result_card, textvariable=self.roas_var, font=("Helvetica", 12, "bold"), fg="#334155")
        self.roas_lbl.pack(anchor="w", pady=(0, 8))

        ttk.Label(result_card, text="💡 行銷策略主管覆核與預算配置指引：", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.advice_txt = tk.Text(result_card, height=13, font=("Helvetica", 9), wrap="word", bg="#f8fafc", relief="solid", borderwidth=1)
        self.advice_txt.pack(fill=tk.BOTH, expand=True)
        self.advice_txt.insert("1.0", "調整左側各通路預算或點擊上方情境，演算法將根據跨通路歷史表現，產出營收預估與行銷預算配比優化建議。")
        self.advice_txt.config(state="disabled")

        self.status_var = tk.StringVar(value="正在初始化機器學習環境與資料集...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w", padding="4")
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def _sync_labels(self):
        self.lbl_meta.config(text=f"${int(self.var_meta.get()):,}")
        self.lbl_goog.config(text=f"${int(self.var_google.get()):,}")
        self.lbl_kol.config(text=f"${int(self.var_kol.get()):,}")
        self.lbl_line.config(text=f"${int(self.var_line.get()):,}")
        self.lbl_trend.config(text=f"{int(self.var_trend.get())} 分")
        self.lbl_coup.config(text=f"{int(self.var_coupon.get()):,} 張")

    def _load_preset_high(self):
        self.var_season.set("雙11大促")
        self.var_meta.set(150000)
        self.var_google.set(160000)
        self.var_kol.set(180000)
        self.var_line.set(28000)
        self.var_trend.set(90)
        self.var_coupon.set(12000)
        self._sync_labels()
        self._predict()

    def _load_preset_medium(self):
        self.var_season.set("年中慶")
        self.var_meta.set(70000)
        self.var_google.set(90000)
        self.var_kol.set(50000)
        self.var_line.set(15000)
        self.var_trend.set(65)
        self.var_coupon.set(4000)
        self._sync_labels()
        self._predict()

    def _load_preset_low(self):
        self.var_season.set("平日檔期")
        self.var_meta.set(20000)
        self.var_google.set(35000)
        self.var_kol.set(0)
        self.var_line.set(5000)
        self.var_trend.set(35)
        self.var_coupon.set(1000)
        self._sync_labels()
        self._predict()

    def _load_random_case(self):
        self.var_season.set(random.choice(["雙11大促", "年中慶", "春節連假", "平日檔期"]))
        self.var_meta.set(random.randint(10000, 160000))
        self.var_google.set(random.randint(10000, 170000))
        self.var_kol.set(random.choice([0, 20000, 50000, 100000, 150000]))
        self.var_line.set(random.randint(2000, 30000))
        self.var_trend.set(random.randint(25, 95))
        self.var_coupon.set(random.randint(800, 10000))
        self._sync_labels()
        self._predict()

    def _predict(self):
        if not self.pipeline:
            messagebox.showwarning("模型未載入", "模型尚未訓練或載入完成！")
            return

        input_data = pd.DataFrame([{
            "Meta臉書廣告_TWD": int(self.var_meta.get()),
            "Google搜尋廣告_TWD": int(self.var_google.get()),
            "KOL網紅行銷_TWD": int(self.var_kol.get()),
            "LINE官方推播_TWD": int(self.var_line.get()),
            "季節檔期": self.var_season.get(),
            "品牌搜尋熱度": int(self.var_trend.get()),
            "促銷折價券發放數": int(self.var_coupon.get())
        }])

        try:
            pred_rev = self.pipeline.predict(input_data)[0]
            total_ads = (int(self.var_meta.get()) + int(self.var_google.get()) +
                         int(self.var_kol.get()) + int(self.var_line.get()))
            roas = pred_rev / total_ads if total_ads > 0 else 0

            self.res_tag_var.set(f"預估營業額：${pred_rev:,.0f} TWD")
            self.roas_var.set(f"總行銷費用：${total_ads:,} ｜ 預估 ROAS：{roas:.2f} 倍")

            advice = (
                f"【行銷主管評估方針：{self.var_season.get()}】\n"
                f"1. 預估營收規模：本波檔期預估可創造約 ${pred_rev:,.0f} 元營收。\n"
                f"2. 廣告效益評估：每投入 1 元廣告費用，預計帶回 {roas:.2f} 元營業額。\n"
                "3. 通路配比優化：Google 搜尋廣告與 LINE 私域推播對直接轉換具有極高邊際效益，建議保持充足預算。\n"
                "4. 庫存與人力調配：請將此營收預估同步提供倉儲出貨與客服團隊，提早備妥包材與客服人力排班。"
            )

            self.advice_txt.config(state="normal")
            self.advice_txt.delete("1.0", tk.END)
            self.advice_txt.insert("1.0", advice)
            self.advice_txt.config(state="disabled")
        except Exception as e:
            messagebox.showerror("預測錯誤", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = AdsRevenueApp(root)
    root.mainloop()
