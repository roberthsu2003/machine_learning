#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智慧物流調度 - 宅配包裹在途配送延遲風險預警系統 (Tkinter 桌面 GUI 實戰版)
------------------------------------------------------------------
功能特點：
1. 完整機器學習全流程：自動載入轉運站負荷與天候路況數據、特徵預處理 Pipeline、隨機森林分類推論。
2. 內建實務物流情境：提供 3 組具代表性的配送情境（偏遠花東暴雨、大促爆倉、都會常態早班）一鍵載入。
3. 互動式介面：起迄區域、出貨時間、天氣下拉選單，與重量、站所總量滑桿即時連動。
4. 即時預警與物流處遇：預測配送超時延遲機率，並產出幹線專車加派、路徑動態重組與預警通報建議。
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
        os.path.join(os.path.dirname(__file__), "logistics.csv"),
        os.path.join(os.getcwd(), "logistics.csv"),
        os.path.join(os.getcwd(), "AI協作機器學習_職場實戰篇", "09_物流_宅配包裹延遲預警", "logistics.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class LogisticsDelayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚚 智慧物流調度 - 宅配包裹配送延遲預警系統")
        self.root.geometry("880x820")
        self.root.minsize(820, 760)

        self.pipeline = None
        self.metrics = {}

        self._build_ui()
        self._train_model()

    def _train_model(self):
        data_path = get_data_path()
        if not data_path:
            self.status_var.set("❌ 錯誤：找不到 logistics.csv 資料集！請確認檔案位置。")
            messagebox.showerror("檔案遺失", "找不到 logistics.csv 資料集，請確保其位於當前專案資料夾！")
            return

        try:
            df = pd.read_csv(data_path)
            X = df.drop(columns=["配送是否延遲"])
            y = (df["配送是否延遲"] == "是").astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            num_cols = ["包裹重量_kg", "當日轉運站包裹總量"]
            cat_cols = ["起運區域", "配送行政區", "包裝材積大小", "出貨時間帶", "降雨強度", "是否為大促檔期"]

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
            self.status_var.set(f"✅ 模型訓練就緒 ｜ 測試集準確率：{acc*100:.1f}% ｜ ROC-AUC：{auc:.3f} ｜ 樣本規模：{len(df):,} 筆貨件")
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
            text="🚚 宅配物流行控中心 — 包裹在途時效與延遲風險預警系統",
            font=("Helvetica", 14, "bold"),
            foreground="#0369a1"
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="依據起訖站所、天氣路況、材積重量與大促爆倉負荷，即時預警配送超時風險並指派調度行動",
            font=("Helvetica", 9),
            foreground="#475569"
        ).pack(anchor="w", pady=(2, 0))

        preset_frame = ttk.LabelFrame(main_frame, text="⚡ 快速物流配送情境 (點擊一鍵帶入)", padding="8")
        preset_frame.pack(fill=tk.X, pady=(0, 10))

        btn_box = ttk.Frame(preset_frame)
        btn_box.pack(fill=tk.X)

        ttk.Button(btn_box, text="🚨 情境 A：高延遲危險件 (花東線/暴雨/夜配/大促)", command=self._load_preset_high).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="⚖️ 情境 B：轉運負擔較重件 (台中至山區/午後中材積)", command=self._load_preset_medium).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🟢 情境 C：都會暢通順利件 (林口至台北市/晨間小件)", command=self._load_preset_low).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🎲 隨機抽樣包裹", command=self._load_random_case).pack(side=tk.LEFT, padx=3)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        input_card = ttk.LabelFrame(content_frame, text="📦 貨件規格與配送環境特徵", padding="12")
        input_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.var_origin = tk.StringVar(value="北部轉運中心(林口)")
        self.var_dest = tk.StringVar(value="花蓮台東花東線")
        self.var_weight = tk.DoubleVar(value=8.5)
        self.var_size = tk.StringVar(value="大材積(91-120cm)")
        self.var_time = tk.StringVar(value="傍晚夜配批次(18:00)")
        self.var_hub = tk.IntVar(value=68000)
        self.var_rain = tk.StringVar(value="大雨滂沱/暴雨")
        self.var_sale = tk.StringVar(value="是")

        r = 0
        ttk.Label(input_card, text="起運轉運節點：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_origin, values=["北部轉運中心(林口)", "中部轉運中心(台中)", "南部轉運中心(高雄)"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="配送目的行政區：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_dest, values=["台北市都會區", "新北市偏遠山區", "台中市都會區", "高雄市都會區", "花蓮台東花東線"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="包裹重量 (kg)：").grid(row=r, column=0, sticky="w", pady=4)
        f_wei = ttk.Frame(input_card)
        f_wei.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_wei, from_=0.5, to=30.0, variable=self.var_weight, command=lambda v: self.lbl_wei.config(text=f"{float(v):.1f} kg")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_wei = ttk.Label(f_wei, text="8.5 kg", width=8, anchor="e")
        self.lbl_wei.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="包裝材積大小：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_size, values=["小材積(60cm以下)", "中材積(61-90cm)", "大材積(91-120cm)", "超大材積(120cm以上)"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="轉運出車時段：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_time, values=["晨間批次(08:00)", "午後批次(13:00)", "傍晚夜配批次(18:00)"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="當日轉運站進件總量：").grid(row=r, column=0, sticky="w", pady=4)
        f_hub = ttk.Frame(input_card)
        f_hub.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_hub, from_=15000, to=85000, variable=self.var_hub, command=lambda v: self.lbl_hub.config(text=f"{int(float(v)):,} 件")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_hub = ttk.Label(f_hub, text="68,000 件", width=10, anchor="e")
        self.lbl_hub.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="沿途降雨路況：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_rain, values=["無雨/晴朗", "局部陣雨", "大雨滂沱/暴雨"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="是否為電商大促檔期：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_sale, values=["是", "否"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        btn_pred = tk.Button(
            input_card,
            text="🚀 啟動機器學習模型評估延遲機率",
            bg="#0284c7",
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief="raised",
            cursor="hand2",
            pady=8,
            command=self._predict
        )
        btn_pred.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(15, 0))

        result_card = ttk.LabelFrame(content_frame, text="📊 延遲風險預測與調度處遇建議", padding="12")
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

        self.prob_var = tk.StringVar(value="預估延遲機率：-- %")
        self.prob_lbl = tk.Label(result_card, textvariable=self.prob_var, font=("Helvetica", 13, "bold"), fg="#1e293b")
        self.prob_lbl.pack(anchor="w", pady=(0, 4))

        self.prob_bar = ttk.Progressbar(result_card, orient="horizontal", mode="determinate", length=200)
        self.prob_bar.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(result_card, text="🚛 行控調度與客服安撫處遇指引：", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.advice_txt = tk.Text(result_card, height=13, font=("Helvetica", 9), wrap="word", bg="#f8fafc", relief="solid", borderwidth=1)
        self.advice_txt.pack(fill=tk.BOTH, expand=True)
        self.advice_txt.insert("1.0", "輸入左側貨件環境特徵或選擇情境，系統將根據路況、天氣與站所負荷數據，精確計算延遲機率並產出行控防範指引。")
        self.advice_txt.config(state="disabled")

        self.status_var = tk.StringVar(value="正在初始化機器學習環境與資料集...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w", padding="4")
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def _sync_labels(self):
        self.lbl_wei.config(text=f"{float(self.var_weight.get()):.1f} kg")
        self.lbl_hub.config(text=f"{int(self.var_hub.get()):,} 件")

    def _load_preset_high(self):
        self.var_origin.set("北部轉運中心(林口)")
        self.var_dest.set("花蓮台東花東線")
        self.var_weight.set(12.0)
        self.var_size.set("超大材積(120cm以上)")
        self.var_time.set("傍晚夜配批次(18:00)")
        self.var_hub.set(75000)
        self.var_rain.set("大雨滂沱/暴雨")
        self.var_sale.set("是")
        self._sync_labels()
        self._predict()

    def _load_preset_medium(self):
        self.var_origin.set("中部轉運中心(台中)")
        self.var_dest.set("新北市偏遠山區")
        self.var_weight.set(5.0)
        self.var_size.set("中材積(61-90cm)")
        self.var_time.set("午後批次(13:00)")
        self.var_hub.set(52000)
        self.var_rain.set("局部陣雨")
        self.var_sale.set("否")
        self._sync_labels()
        self._predict()

    def _load_preset_low(self):
        self.var_origin.set("北部轉運中心(林口)")
        self.var_dest.set("台北市都會區")
        self.var_weight.set(1.5)
        self.var_size.set("小材積(60cm以下)")
        self.var_time.set("晨間批次(08:00)")
        self.var_hub.set(32000)
        self.var_rain.set("無雨/晴朗")
        self.var_sale.set("否")
        self._sync_labels()
        self._predict()

    def _load_random_case(self):
        self.var_origin.set(random.choice(["北部轉運中心(林口)", "中部轉運中心(台中)", "南部轉運中心(高雄)"]))
        self.var_dest.set(random.choice(["台北市都會區", "新北市偏遠山區", "台中市都會區", "高雄市都會區", "花蓮台東花東線"]))
        self.var_weight.set(round(random.uniform(0.5, 20.0), 1))
        self.var_size.set(random.choice(["小材積(60cm以下)", "中材積(61-90cm)", "大材積(91-120cm)", "超大材積(120cm以上)"]))
        self.var_time.set(random.choice(["晨間批次(08:00)", "午後批次(13:00)", "傍晚夜配批次(18:00)"]))
        self.var_hub.set(random.randint(20000, 80000))
        self.var_rain.set(random.choice(["無雨/晴朗", "局部陣雨", "大雨滂沱/暴雨"]))
        self.var_sale.set(random.choice(["是", "否"]))
        self._sync_labels()
        self._predict()

    def _predict(self):
        if not self.pipeline:
            messagebox.showwarning("模型未載入", "模型尚未訓練或載入完成！")
            return

        input_data = pd.DataFrame([{
            "起運區域": self.var_origin.get(),
            "配送行政區": self.var_dest.get(),
            "包裹重量_kg": float(self.var_weight.get()),
            "包裝材積大小": self.var_size.get(),
            "出貨時間帶": self.var_time.get(),
            "當日轉運站包裹總量": int(self.var_hub.get()),
            "降雨強度": self.var_rain.get(),
            "是否為大促檔期": self.var_sale.get()
        }])

        try:
            prob = self.pipeline.predict_proba(input_data)[0][1]
            pct = prob * 100
            self.prob_bar["value"] = pct
            self.prob_var.set(f"預估配送延遲機率：{pct:.1f} %")

            if prob >= 0.65:
                self.res_tag_var.set("🚨 極高超時延遲風險 (需即刻調度介入)")
                self.res_tag_lbl.config(fg="#b91c1c", bg="#fee2e2")
                advice = (
                    "【物流調度處遇建議：高危延遲警報】\n"
                    "1. 綠色通道提早發車：受天候暴雨與偏遠路線影響，建議幹線司機提前 45 分鐘出車避開尖峰塞車。\n"
                    "2. 分流加派專車：該站所已達爆倉警戒，建議調用第三方外包貨車分擔 20% 配送量。\n"
                    "3. 顧客主動預警簡訊：系統自動觸發簡訊告知收件人『受氣候影響稍有延遲』，大幅降低 70% 客訴電話。"
                )
            elif prob >= 0.35:
                self.res_tag_var.set("⚖️ 中度延遲傾向 (密切監控路況)")
                self.res_tag_lbl.config(fg="#b45309", bg="#fef3c7")
                advice = (
                    "【物流調度處遇建議：動態監控】\n"
                    "1. 建議配送司機利用即時導航路徑避開施工路段。\n"
                    "2. 提醒司機於送達前半小時電話聯繫收件人確認在家，提高初次妥投成功率。\n"
                    "3. 追蹤該站所午後發車進度，確保末班件在法規時效內妥投。"
                )
            else:
                self.res_tag_var.set("🟢 配送時效優良 (可如期在時限內送達)")
                self.res_tag_lbl.config(fg="#15803d", bg="#dcfce7")
                advice = (
                    "【物流調度處遇建議：標準順暢流程】\n"
                    "1. 都會路況良好且站所處理餘裕充足，依標準排程出車。\n"
                    "2. 預計可在承諾時段前半段順暢妥投，維持優良服務體驗 (SLA)。"
                )

            self.advice_txt.config(state="normal")
            self.advice_txt.delete("1.0", tk.END)
            self.advice_txt.insert("1.0", advice)
            self.advice_txt.config(state="disabled")
        except Exception as e:
            messagebox.showerror("預測錯誤", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = LogisticsDelayApp(root)
    root.mainloop()
