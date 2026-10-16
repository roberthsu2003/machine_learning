#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手電商選品 - 3C 精品二手估價與建議售價系統 (Tkinter 桌面 GUI 實戰版)
------------------------------------------------------------------
功能特點：
1. 完整機器學習全流程：自動載入二手歷史交易數據、折舊估算 Pipeline、隨機森林迴歸模型即時推論。
2. 內建實務估價情境：提供 3 組具代表性的二手成色情境（近全新保固內、一般使用常規機、老舊無配件機）一鍵試算。
3. 互動式介面：品類、成色、配件、品牌下拉選單，與原價、使用年數、剩餘保固月數數值調整。
4. 即時估價與定價獲利建議：預估合理二手售價，並產出收購行情與上架利潤空間建議。
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
        os.path.join(os.path.dirname(__file__), "used_items.csv"),
        os.path.join(os.getcwd(), "used_items.csv"),
        os.path.join(os.getcwd(), "AI協作機器學習_職場實戰篇", "04_選品_二手商品定價估算", "used_items.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class UsedPriceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🏷️ 二手選品與電商 - 二手商品估價與定價建議系統")
        self.root.geometry("880x820")
        self.root.minsize(820, 760)

        self.pipeline = None
        self.metrics = {}

        self._build_ui()
        self._train_model()

    def _train_model(self):
        data_path = get_data_path()
        if not data_path:
            self.status_var.set("❌ 錯誤：找不到 used_items.csv 資料集！請確認檔案位置。")
            messagebox.showerror("檔案遺失", "找不到 used_items.csv 資料集，請確保其位於當前專案資料夾！")
            return

        try:
            df = pd.read_csv(data_path)
            X = df.drop(columns=["二手建議售價_TWD"])
            y = df["二手建議售價_TWD"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            num_cols = ["原價_TWD", "使用年限_年", "原廠保固剩餘月數"]
            cat_cols = ["品類名稱", "商品成色", "配件完整度", "品牌等級"]

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
            text="📱 二手電商與選品賣家 — 二手商品成色折舊與智能估價系統 (迴歸篇)",
            font=("Helvetica", 14, "bold"),
            foreground="#c2410c"
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="根據商品品類、成色外觀、使用年限與保固配件，機器學習即時估算市場行情與建議上架售價",
            font=("Helvetica", 9),
            foreground="#475569"
        ).pack(anchor="w", pady=(2, 0))

        preset_frame = ttk.LabelFrame(main_frame, text="⚡ 快速估價範例情境 (點擊一鍵帶入)", padding="8")
        preset_frame.pack(fill=tk.X, pady=(0, 10))

        btn_box = ttk.Frame(preset_frame)
        btn_box.pack(fill=tk.X)

        ttk.Button(btn_box, text="✨ 情境 A：高殘值極品 (99新旗艦機/盒裝保固內)", command=self._load_preset_high).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="⚖️ 情境 B：常態二手良品 (85新使用兩年/配件齊)", command=self._load_preset_medium).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="📦 情境 C：殘值機出清 (明顯痕跡/無盒裝過保)", command=self._load_preset_low).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🎲 隨機產生商品", command=self._load_random_case).pack(side=tk.LEFT, padx=3)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        input_card = ttk.LabelFrame(content_frame, text="🔍 商品規格與成色條件設定", padding="12")
        input_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.var_category = tk.StringVar(value="iPhone手機")
        self.var_orig_price = tk.IntVar(value=36000)
        self.var_condition = tk.StringVar(value="9成新(輕微痕跡)")
        self.var_years = tk.DoubleVar(value=1.5)
        self.var_accessories = tk.StringVar(value="盒裝配件齊全")
        self.var_brand_tier = tk.StringVar(value="一線國際旗艦")
        self.var_warranty = tk.IntVar(value=6)

        r = 0
        ttk.Label(input_card, text="品類名稱：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_category, values=["iPhone手機", "電競筆電", "數位單眼相機", "機械腕錶", "iPad平板"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="新品購入原價 (TWD)：").grid(row=r, column=0, sticky="w", pady=4)
        f_orig = ttk.Frame(input_card)
        f_orig.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_orig, from_=10000, to=100000, variable=self.var_orig_price, command=lambda v: self.lbl_orig.config(text=f"${int(float(v)):,}")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_orig = ttk.Label(f_orig, text="$36,000", width=10, anchor="e")
        self.lbl_orig.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="商品成色等級：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_condition, values=["99成新(僅拆封)", "9成新(輕微痕跡)", "8成新(正常使用)", "7成新(明顯磨損)"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="已使用時間 (年)：").grid(row=r, column=0, sticky="w", pady=4)
        f_yr = ttk.Frame(input_card)
        f_yr.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_yr, from_=0.1, to=6.0, variable=self.var_years, command=lambda v: self.lbl_yr.config(text=f"{float(v):.1f} 年")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_yr = ttk.Label(f_yr, text="1.5 年", width=8, anchor="e")
        self.lbl_yr.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="配件完整度：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_accessories, values=["盒裝配件齊全", "僅單機附副廠線", "無任何配件"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="品牌等級：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_brand_tier, values=["一線國際旗艦", "主流知名大廠", "新興平價品牌"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="原廠保固剩餘月數：").grid(row=r, column=0, sticky="w", pady=4)
        f_war = ttk.Frame(input_card)
        f_war.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_war, from_=0, to=24, variable=self.var_warranty, command=lambda v: self.lbl_war.config(text=f"{int(float(v))} 個月")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_war = ttk.Label(f_war, text="6 個月", width=8, anchor="e")
        self.lbl_war.pack(side=tk.RIGHT)

        r += 1
        btn_pred = tk.Button(
            input_card,
            text="🚀 啟動機器學習模型評估二手建議售價",
            bg="#ea580c",
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief="raised",
            cursor="hand2",
            pady=8,
            command=self._predict
        )
        btn_pred.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(15, 0))

        result_card = ttk.LabelFrame(content_frame, text="📊 估價評估與選品收購獲利建議", padding="12")
        result_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        self.res_tag_var = tk.StringVar(value="尚未執行估價")
        self.res_tag_lbl = tk.Label(
            result_card,
            textvariable=self.res_tag_var,
            font=("Helvetica", 18, "bold"),
            fg="#c2410c",
            bg="#fff7ed",
            pady=12,
            relief="groove"
        )
        self.res_tag_lbl.pack(fill=tk.X, pady=(0, 10))

        self.margin_var = tk.StringVar(value="建議二手收購行情區間：-- 元")
        self.margin_lbl = tk.Label(result_card, textvariable=self.margin_var, font=("Helvetica", 12, "bold"), fg="#334155")
        self.margin_lbl.pack(anchor="w", pady=(0, 8))

        ttk.Label(result_card, text="💡 二手店家上架與利潤保證策略：", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.advice_txt = tk.Text(result_card, height=13, font=("Helvetica", 9), wrap="word", bg="#f8fafc", relief="solid", borderwidth=1)
        self.advice_txt.pack(fill=tk.BOTH, expand=True)
        self.advice_txt.insert("1.0", "輸入左側商品成色或點擊上方情境，系統將根據歷年 1,500 筆二手實價交易數據，即時推估市場行情與定價策略。")
        self.advice_txt.config(state="disabled")

        self.status_var = tk.StringVar(value="正在初始化機器學習環境與資料集...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w", padding="4")
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def _sync_labels(self):
        self.lbl_orig.config(text=f"${int(self.var_orig_price.get()):,}")
        self.lbl_yr.config(text=f"{float(self.var_years.get()):.1f} 年")
        self.lbl_war.config(text=f"{int(self.var_warranty.get())} 個月")

    def _load_preset_high(self):
        self.var_category.set("iPhone手機")
        self.var_orig_price.set(38900)
        self.var_condition.set("99成新(僅拆封)")
        self.var_years.set(0.3)
        self.var_accessories.set("盒裝配件齊全")
        self.var_brand_tier.set("一線國際旗艦")
        self.var_warranty.set(10)
        self._sync_labels()
        self._predict()

    def _load_preset_medium(self):
        self.var_category.set("電競筆電")
        self.var_orig_price.set(45000)
        self.var_condition.set("8成新(正常使用)")
        self.var_years.set(2.0)
        self.var_accessories.set("盒裝配件齊全")
        self.var_brand_tier.set("主流知名大廠")
        self.var_warranty.set(0)
        self._sync_labels()
        self._predict()

    def _load_preset_low(self):
        self.var_category.set("iPad平板")
        self.var_orig_price.set(22000)
        self.var_condition.set("7成新(明顯磨損)")
        self.var_years.set(4.0)
        self.var_accessories.set("無任何配件")
        self.var_brand_tier.set("主流知名大廠")
        self.var_warranty.set(0)
        self._sync_labels()
        self._predict()

    def _load_random_case(self):
        self.var_category.set(random.choice(["iPhone手機", "電競筆電", "數位單眼相機", "機械腕錶", "iPad平板"]))
        self.var_orig_price.set(random.choice([18000, 25000, 36000, 48000, 68000]))
        self.var_condition.set(random.choice(["99成新(僅拆封)", "9成新(輕微痕跡)", "8成新(正常使用)", "7成新(明顯磨損)"]))
        self.var_years.set(round(random.uniform(0.2, 5.0), 1))
        self.var_accessories.set(random.choice(["盒裝配件齊全", "僅單機附副廠線", "無任何配件"]))
        self.var_brand_tier.set(random.choice(["一線國際旗艦", "主流知名大廠", "新興平價品牌"]))
        self.var_warranty.set(random.choice([0, 3, 6, 12]))
        self._sync_labels()
        self._predict()

    def _predict(self):
        if not self.pipeline:
            messagebox.showwarning("模型未載入", "模型尚未訓練或載入完成！")
            return

        input_data = pd.DataFrame([{
            "品類名稱": self.var_category.get(),
            "原價_TWD": int(self.var_orig_price.get()),
            "商品成色": self.var_condition.get(),
            "使用年限_年": float(self.var_years.get()),
            "配件完整度": self.var_accessories.get(),
            "品牌等級": self.var_brand_tier.get(),
            "原廠保固剩餘月數": int(self.var_warranty.get())
        }])

        try:
            pred_price = self.pipeline.predict(input_data)[0]
            buy_price_low = pred_price * 0.70
            buy_price_high = pred_price * 0.78
            orig_p = int(self.var_orig_price.get())
            retained_val = (pred_price / orig_p) * 100

            self.res_tag_var.set(f"建議二手售價：${pred_price:,.0f} TWD")
            self.margin_var.set(f"店家收購建議價：${buy_price_low:,.0f} ~ ${buy_price_high:,.0f} (殘值率 {retained_val:.1f}%)")

            advice = (
                f"【二手選品獲利分析：{self.var_category.get()} ({self.var_condition.get()})】\n"
                f"1. 市場參考售價：預估終端買家合理接盤價為 ${pred_price:,.0f} 元。\n"
                f"2. 店家收購安全區：建議以 ${buy_price_low:,.0f} ~ ${buy_price_high:,.0f} 元收購，以確保 22%~30% 翻新毛利空間。\n"
                "3. 上架加值技巧：若為『盒裝齊全且有原廠保固』，可在拍賣平台標題突顯『公司貨在保』，通常能加速 50% 售出週轉期。\n"
                "4. 庫存週轉警訊：若上架超過 30 天未成交，建議分段降價 8%，避免 3C 產品跌價折舊風險。"
            )

            self.advice_txt.config(state="normal")
            self.advice_txt.delete("1.0", tk.END)
            self.advice_txt.insert("1.0", advice)
            self.advice_txt.config(state="disabled")
        except Exception as e:
            messagebox.showerror("預測錯誤", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = UsedPriceApp(root)
    root.mainloop()
