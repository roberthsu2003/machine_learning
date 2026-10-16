#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
廣告效益預測系統 - 桌面 GUI 應用程式 (Tkinter 實戰版)
--------------------------------------------------------
功能特點：
1. 完整機器學習全流程：自動載入資料、特徵預處理 Pipeline、模型訓練與即時推論。
2. 內建預測範例資料：解決使用者/學生沒有自備預測資料的問題，提供 3 組實務情境一鍵載入與隨機情境產生。
3. 互動式介面：支援滑桿與數值輸入框雙向同步，直觀調控行銷預算。
4. 即時預測與效益回饋：預測未來銷售額並計算總預算與投報效益。
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_data_path():
    """自動尋找 advSale.csv 檔案路徑"""
    candidates = [
        os.path.join(os.path.dirname(__file__), "advSale.csv"),
        os.path.join(os.getcwd(), "advSale.csv"),
        os.path.join(os.getcwd(), "AI提問與機器學習實戰", "01_廣告效益預測_迴歸篇", "advSale.csv"),
        os.path.join(os.path.dirname(__file__), "data", "advSale.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None



class AdPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📈 廣告效益預測系統 - 機器學習迴歸應用")
        self.root.geometry("820x760")
        self.root.minsize(760, 700)

        # 預設字型設定
        self.default_font = ("Helvetica", 11)
        self.title_font = ("Helvetica", 14, "bold")

        # 核心 ML 變數
        self.model_pipeline = None
        self.metrics = {}

        # 建立 UI
        self._build_ui()

        # 啟動時訓練模型
        self._train_model()

    def _build_ui(self):
        # 主外框
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. 標題與模型狀態
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        title_lbl = ttk.Label(
            header_frame,
            text="📊 廣告效益與產品銷售額預測系統",
            font=("Helvetica", 16, "bold"),
            foreground="#1e3a8a",
        )
        title_lbl.pack(anchor=tk.W)

        subtitle_lbl = ttk.Label(
            header_frame,
            text="使用 Scikit-Learn 迴歸 Pipeline 演算法 ｜ 支援即時預算模擬與實務情境預測",
            font=("Helvetica", 9),
            foreground="#64748b",
        )
        subtitle_lbl.pack(anchor=tk.W, pady=(2, 0))

        # 模型效能標籤面板
        self.status_var = tk.StringVar(value="⏳ 模型訓練中...")
        self.status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            font=("Helvetica", 9),
            background="#f1f5f9",
            padding=6,
        )
        self.status_bar.pack(fill=tk.X, pady=(0, 12))

        # 2. 快速載入預設預測資料專區 (針對學生無預測資料的貼心設計)
        preset_lf = ttk.LabelFrame(main_frame, text="💡 快速預測測試範例 (點擊一鍵套用投放情境)", padding=10)
        preset_lf.pack(fill=tk.X, pady=(0, 12))

        btn_box = ttk.Frame(preset_lf)
        btn_box.pack(fill=tk.X)

        btn_a = ttk.Button(
            btn_box,
            text="🌟 情境 A：電視主打重磅曝光",
            command=lambda: self.load_preset(92.0, 26.0, 5.0, "大於百萬"),
        )
        btn_a.pack(side=tk.LEFT, padx=(0, 6), expand=True, fill=tk.X)

        btn_b = ttk.Button(
            btn_box,
            text="⚖️ 情境 B：全渠道平衡推廣",
            command=lambda: self.load_preset(55.0, 18.0, 3.5, "十萬到百萬"),
        )
        btn_b.pack(side=tk.LEFT, padx=6, expand=True, fill=tk.X)

        btn_c = ttk.Button(
            btn_box,
            text="📱 情境 C：社群自媒體新創",
            command=lambda: self.load_preset(15.0, 6.0, 9.0, "一萬到十萬"),
        )
        btn_c.pack(side=tk.LEFT, padx=6, expand=True, fill=tk.X)

        btn_random = ttk.Button(
            btn_box,
            text="🎲 隨機情境",
            command=self.load_random_scenario,
        )
        btn_random.pack(side=tk.LEFT, padx=(6, 0), expand=True, fill=tk.X)

        # 3. 廣告預算參數手動調整區
        input_lf = ttk.LabelFrame(main_frame, text="⚙️ 廣告預算參數手動配置", padding=12)
        input_lf.pack(fill=tk.X, pady=(0, 12))

        # 電視廣告 (10 ~ 100 千元)
        self.tv_var = tk.DoubleVar(value=50.0)
        self._create_slider_row(input_lf, "電視廣告預算 (千元):", self.tv_var, 10.0, 100.0, 0)

        # 廣播廣告 (0 ~ 50 千元)
        self.radio_var = tk.DoubleVar(value=18.0)
        self._create_slider_row(input_lf, "廣播廣告預算 (千元):", self.radio_var, 0.0, 50.0, 1)

        # 社群媒體廣告 (0 ~ 15 千元)
        self.social_var = tk.DoubleVar(value=3.0)
        self._create_slider_row(input_lf, "社群廣告預算 (千元):", self.social_var, 0.0, 15.0, 2)

        # 網紅合作級距 (下拉選單)
        ttk.Label(input_lf, text="網紅合作級距:", font=self.default_font).grid(
            row=3, column=0, sticky=tk.W, pady=8
        )
        self.influencer_var = tk.StringVar(value="十萬到百萬")
        self.influencer_cb = ttk.Combobox(
            input_lf,
            textvariable=self.influencer_var,
            values=["大於百萬", "十萬到百萬", "一萬到十萬", "小於一萬"],
            state="readonly",
            width=22,
        )
        self.influencer_cb.grid(row=3, column=1, sticky=tk.W, pady=8)

        # 4. 操作按鈕區
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=(4, 12))

        predict_btn = tk.Button(
            action_frame,
            text="🚀 計算預期產品銷售額",
            font=("Helvetica", 12, "bold"),
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            padx=12,
            pady=8,
            relief=tk.FLAT,
            command=self.predict_sales,
        )
        predict_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        reset_btn = ttk.Button(
            action_frame,
            text="🔄 重設預設值",
            command=lambda: self.load_preset(50.0, 18.0, 3.0, "十萬到百萬"),
        )
        reset_btn.pack(side=tk.RIGHT, padx=(6, 0), ipady=6)

        # 5. 預測結果展示卡片
        result_lf = ttk.LabelFrame(main_frame, text="📊 模型預測結果與商業決策回饋", padding=12)
        result_lf.pack(fill=tk.BOTH, expand=True)

        self.res_sales_var = tk.StringVar(value="點擊上方按鈕開始預測")
        sales_banner = ttk.Label(
            result_lf,
            textvariable=self.res_sales_var,
            font=("Helvetica", 18, "bold"),
            foreground="#0f766e",
            anchor=tk.CENTER,
            background="#ccfbf1",
            padding=10,
        )
        sales_banner.pack(fill=tk.X, pady=(0, 10))

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
            "1. 點擊上方的「快速預測測試範例」按鈕，即可自動帶入實務行銷數據。\n"
            "2. 您也可以透過滑桿自訂不同管道的預算投入。\n"
            "3. 點擊「計算預期產品銷售額」，系統將透過已訓練完成的機器學習模型即時推論！"
        )
        self.res_detail_text.config(state=tk.DISABLED)

    def _create_slider_row(self, parent, label_text, var, min_v, max_v, row_idx):
        """建立滑桿與文字框聯動的輸入行"""
        ttk.Label(parent, text=label_text, font=self.default_font).grid(
            row=row_idx, column=0, sticky=tk.W, pady=6
        )

        slider = ttk.Scale(
            parent,
            from_=min_v,
            to=max_v,
            variable=var,
            orient=tk.HORIZONTAL,
            length=320,
            command=lambda v: entry_var.set(f"{float(v):.1f}"),
        )
        slider.grid(row=row_idx, column=1, padx=10, pady=6, sticky=tk.W)

        entry_var = tk.StringVar(value=f"{var.get():.1f}")
        entry = ttk.Entry(parent, textvariable=entry_var, width=8)
        entry.grid(row=row_idx, column=2, padx=5, pady=6)

        def on_entry_change(*_):
            try:
                val = float(entry_var.get())
                if min_v <= val <= max_v:
                    var.set(val)
            except ValueError:
                pass

        entry_var.trace_add("write", on_entry_change)

    def load_preset(self, tv, radio, social, influencer):
        """載入預設情境資料"""
        self.tv_var.set(tv)
        self.radio_var.set(radio)
        self.social_var.set(social)
        self.influencer_var.set(influencer)
        self.predict_sales()

    def load_random_scenario(self):
        """隨機產生合理的廣告投放測試資料"""
        tv = round(random.uniform(10.0, 98.0), 1)
        radio = round(random.uniform(1.0, 45.0), 1)
        social = round(random.uniform(0.5, 12.0), 1)
        influencer = random.choice(["大於百萬", "十萬到百萬", "一萬到十萬", "小於一萬"])
        self.load_preset(tv, radio, social, influencer)

    def _train_model(self):
        """背景自動讀取資料集並訓練線性迴歸 Pipeline"""
        data_path = get_data_path()
        if not data_path:
            self.status_var.set("❌ 錯誤：找不到 data/advSale.csv，請確認資料路徑！")
            messagebox.showerror("路徑錯誤", "找不到 data/advSale.csv 檔案！請確保檔案放置於 data 目錄下。")
            return

        try:
            df = pd.read_csv(data_path)
            num_cols = ["電視廣告", "廣播廣告", "社群媒體廣告"]
            cat_cols = ["網紅廣告"]
            target_col = "銷售額"

            X = df[num_cols + cat_cols]
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # 建構前處理器
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num_cols),
                    ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
                ]
            )

            # 迴歸 Pipeline
            self.model_pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", LinearRegression()),
                ]
            )

            # 訓練
            self.model_pipeline.fit(X_train, y_train)

            # 評估
            y_pred = self.model_pipeline.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            self.metrics = {"r2": r2, "mae": mae, "rmse": rmse, "samples": len(df)}
            self.status_var.set(
                f"✅ 模型訓練完成！資料筆數: {len(df):,} 筆 ｜ 測試集 R²: {r2:.4f} ｜ MAE: {mae:.2f} ｜ RMSE: {rmse:.2f}"
            )

            # 自動觸發一次初始預測展示
            self.predict_sales()

        except Exception as e:
            self.status_var.set(f"❌ 訓練失敗: {str(e)}")
            messagebox.showerror("訓練錯誤", f"模型訓練過程中發生錯誤：\n{str(e)}")

    def predict_sales(self):
        """執行銷售額預測並產出決策建議"""
        if self.model_pipeline is None:
            messagebox.showwarning("提示", "模型尚未訓練完成，請稍候！")
            return

        tv = self.tv_var.get()
        radio = self.radio_var.get()
        social = self.social_var.get()
        influencer = self.influencer_var.get()

        # 組裝推論用 DataFrame (維持與訓練時相同的欄位結構與名稱)
        input_df = pd.DataFrame(
            [
                {
                    "電視廣告": tv,
                    "廣播廣告": radio,
                    "社群媒體廣告": social,
                    "網紅廣告": influencer,
                }
            ]
        )

        try:
            pred_sale = self.model_pipeline.predict(input_df)[0]
            total_budget_k = tv + radio + social  # 總投放千元
            total_budget_wan = total_budget_k / 10.0  # 轉為萬元

            # 更新銷售額 Banner
            self.res_sales_var.set(f"預估產品銷售額：{pred_sale:.2f} 萬元")

            # 更新文字解說區
            self.res_detail_text.config(state=tk.NORMAL)
            self.res_detail_text.delete("1.0", tk.END)

            report = (
                f"【預測參數清單】\n"
                f"  • 電視廣告預算：{tv:.1f} 千元\n"
                f"  • 廣播廣告預算：{radio:.1f} 千元\n"
                f"  • 社群廣告預算：{social:.1f} 千元\n"
                f"  • 網紅合作級距：{influencer}\n"
                f"--------------------------------------------------\n"
                f"【行銷投報 (ROI) 效益分析】\n"
                f"  • 總廣告預算投入：約 {total_budget_wan:.2f} 萬元 (即 {total_budget_k:.1f} 千元)\n"
                f"  • 預估達成銷售額：約 {pred_sale:.2f} 萬元\n"
                f"  • 投入產出比例 (銷售額/廣告費用)：約 {pred_sale / max(total_budget_wan, 0.01):.2f} 倍\n"
                f"  • 權重影響觀察：電視廣告對銷售額影響最顯著，搭配高粉絲級距網紅可進一步放大曝光轉換！"
            )

            self.res_detail_text.insert(tk.END, report)
            self.res_detail_text.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("預測錯誤", f"推論過程發生錯誤：\n{str(e)}")


def main():
    root = tk.Tk()
    app = AdPredictionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
