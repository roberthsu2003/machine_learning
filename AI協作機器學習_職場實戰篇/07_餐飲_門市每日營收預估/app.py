#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
連鎖餐飲營運 - 門市每日營收預估與智慧備料排班系統 (Tkinter 桌面 GUI 實戰版)
------------------------------------------------------------------
功能特點：
1. 完整機器學習全流程：自動載入天候商圈歷史營收、特徵預處理 Pipeline、隨機森林迴歸模型推論。
2. 內建實務餐飲情境：提供 3 組具代表性的營業情境（週末大促滿座、平日雨天淡季、商圈大活動）一鍵載入。
3. 互動式介面：商圈、天氣、星期下拉選單，與氣溫、活動熱度、客席訂位率數值調整。
4. 即時預估與內外場處遇：預估當日營業額，並產出生鮮備料進貨量與內外場兼職排班建議。
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
        os.path.join(os.path.dirname(__file__), "store_revenue.csv"),
        os.path.join(os.getcwd(), "store_revenue.csv"),
        os.path.join(os.getcwd(), "AI協作機器學習_職場實戰篇", "07_餐飲_門市每日營收預估", "store_revenue.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class RestaurantRevenueApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🍽️ 連鎖餐飲門市 - 每日營收預估與智慧排班系統")
        self.root.geometry("880x820")
        self.root.minsize(820, 760)

        self.pipeline = None
        self.metrics = {}

        self._build_ui()
        self._train_model()

    def _train_model(self):
        data_path = get_data_path()
        if not data_path:
            self.status_var.set("❌ 錯誤：找不到 store_revenue.csv 資料集！請確認檔案位置。")
            messagebox.showerror("檔案遺失", "找不到 store_revenue.csv 資料集，請確保其位於當前專案資料夾！")
            return

        try:
            df = pd.read_csv(data_path)
            X = df.drop(columns=["每日營收_TWD"])
            y = df["每日營收_TWD"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            num_cols = ["最高氣溫_度C", "商圈活動熱度", "當日訂位客席率"]
            cat_cols = ["商圈型態", "星期別", "是否為國定假日", "天氣狀況"]

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
            self.status_var.set(f"✅ 模型訓練就緒 ｜ 測試集解釋力 R²：{r2:.3f} ｜ 平均絕對誤差 MAE：${mae:,.0f} ｜ 樣本規模：{len(df):,} 天")
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
            text="🍴 餐飲門市營運主管 — 每日營收預估與備料產能規劃系統 (迴歸篇)",
            font=("Helvetica", 14, "bold"),
            foreground="#b45309"
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="依據商圈熱度、天氣狀況、星期別與客席訂位率，機器學習精準預測今日營業額與人力排班需求",
            font=("Helvetica", 9),
            foreground="#475569"
        ).pack(anchor="w", pady=(2, 0))

        preset_frame = ttk.LabelFrame(main_frame, text="⚡ 快速營業情境 (點擊一鍵帶入)", padding="8")
        preset_frame.pack(fill=tk.X, pady=(0, 10))

        btn_box = ttk.Frame(preset_frame)
        btn_box.pack(fill=tk.X)

        ttk.Button(btn_box, text="🔥 情境 A：週末黃金大爆滿 (信義區/假日/好天氣/高訂位)", command=self._load_preset_high).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🌧️ 情境 B：平日雨天淡季 (住宅區/週二/大雨/低訂位)", command=self._load_preset_low).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="⚖️ 情境 C：商業區常態午晚餐 (內科/週五/多雲/訂位7成)", command=self._load_preset_medium).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🎲 隨機排班情境", command=self._load_random_case).pack(side=tk.LEFT, padx=3)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        input_card = ttk.LabelFrame(content_frame, text="📅 當日營業環境與訂位設定", padding="12")
        input_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.var_dist = tk.StringVar(value="信義商務區")
        self.var_weekday = tk.StringVar(value="星期六")
        self.var_holiday = tk.StringVar(value="是")
        self.var_weather = tk.StringVar(value="晴朗多雲")
        self.var_temp = tk.DoubleVar(value=28.0)
        self.var_event = tk.IntVar(value=80)
        self.var_booking = tk.DoubleVar(value=0.85)

        r = 0
        ttk.Label(input_card, text="門市商圈型態：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_dist, values=["信義商務區", "西門觀光區", "板橋住宅區", "內科園區"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="營業星期別：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_weekday, values=["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="是否為國定連假：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_holiday, values=["是", "否"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="天氣狀況預報：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_weather, values=["晴朗多雲", "晴朗炎熱", "局部陣雨", "大雨滂沱"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="最高氣溫 (度C)：").grid(row=r, column=0, sticky="w", pady=4)
        f_tmp = ttk.Frame(input_card)
        f_tmp.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_tmp, from_=12, to=38, variable=self.var_temp, command=lambda v: self.lbl_tmp.config(text=f"{float(v):.1f} °C")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_tmp = ttk.Label(f_tmp, text="28.0 °C", width=8, anchor="e")
        self.lbl_tmp.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="周邊活動熱度指數：").grid(row=r, column=0, sticky="w", pady=4)
        f_eve = ttk.Frame(input_card)
        f_eve.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_eve, from_=10, to=100, variable=self.var_event, command=lambda v: self.lbl_eve.config(text=f"{int(float(v))} 分")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_eve = ttk.Label(f_eve, text="80 分", width=8, anchor="e")
        self.lbl_eve.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="預約訂位客席率：").grid(row=r, column=0, sticky="w", pady=4)
        f_bok = ttk.Frame(input_card)
        f_bok.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_bok, from_=0.1, to=1.0, variable=self.var_booking, command=lambda v: self.lbl_bok.config(text=f"{float(v)*100:.0f}%")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_bok = ttk.Label(f_bok, text="85%", width=8, anchor="e")
        self.lbl_bok.pack(side=tk.RIGHT)

        r += 1
        btn_pred = tk.Button(
            input_card,
            text="🚀 啟動迴歸模型預估今日門市營業額",
            bg="#b45309",
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief="raised",
            cursor="hand2",
            pady=8,
            command=self._predict
        )
        btn_pred.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(15, 0))

        result_card = ttk.LabelFrame(content_frame, text="📊 營收預估與門市排班備料決策", padding="12")
        result_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        self.res_tag_var = tk.StringVar(value="尚未執行預估")
        self.res_tag_lbl = tk.Label(
            result_card,
            textvariable=self.res_tag_var,
            font=("Helvetica", 18, "bold"),
            fg="#b45309",
            bg="#fffbeb",
            pady=12,
            relief="groove"
        )
        self.res_tag_lbl.pack(fill=tk.X, pady=(0, 10))

        self.cap_var = tk.StringVar(value="預估客流規模：-- 席次")
        self.cap_lbl = tk.Label(result_card, textvariable=self.cap_var, font=("Helvetica", 12, "bold"), fg="#334155")
        self.cap_lbl.pack(anchor="w", pady=(0, 8))

        ttk.Label(result_card, text="💡 店長今日營運、備料與排班處遇：", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.advice_txt = tk.Text(result_card, height=13, font=("Helvetica", 9), wrap="word", bg="#f8fafc", relief="solid", borderwidth=1)
        self.advice_txt.pack(fill=tk.BOTH, expand=True)
        self.advice_txt.insert("1.0", "調整左側條件或選擇上方營業情境，系統將根據天候與商圈數據，即時推估營業額並產出精確生鮮進貨量與外場打工排班指引。")
        self.advice_txt.config(state="disabled")

        self.status_var = tk.StringVar(value="正在初始化機器學習環境與資料集...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w", padding="4")
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def _sync_labels(self):
        self.lbl_tmp.config(text=f"{float(self.var_temp.get()):.1f} °C")
        self.lbl_eve.config(text=f"{int(self.var_event.get())} 分")
        self.lbl_bok.config(text=f"{float(self.var_booking.get())*100:.0f}%")

    def _load_preset_high(self):
        self.var_dist.set("信義商務區")
        self.var_weekday.set("星期六")
        self.var_holiday.set("是")
        self.var_weather.set("晴朗多雲")
        self.var_temp.set(27.0)
        self.var_event.set(90)
        self.var_booking.set(0.92)
        self._sync_labels()
        self._predict()

    def _load_preset_low(self):
        self.var_dist.set("板橋住宅區")
        self.var_weekday.set("星期二")
        self.var_holiday.set("否")
        self.var_weather.set("大雨滂沱")
        self.var_temp.set(18.0)
        self.var_event.set(20)
        self.var_booking.set(0.25)
        self._sync_labels()
        self._predict()

    def _load_preset_medium(self):
        self.var_dist.set("內科園區")
        self.var_weekday.set("星期五")
        self.var_holiday.set("否")
        self.var_weather.set("晴朗多雲")
        self.var_temp.set(26.0)
        self.var_event.set(55)
        self.var_booking.set(0.70)
        self._sync_labels()
        self._predict()

    def _load_random_case(self):
        self.var_dist.set(random.choice(["信義商務區", "西門觀光區", "板橋住宅區", "內科園區"]))
        self.var_weekday.set(random.choice(["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]))
        self.var_holiday.set(random.choice(["是", "否"]))
        self.var_weather.set(random.choice(["晴朗多雲", "晴朗炎熱", "局部陣雨", "大雨滂沱"]))
        self.var_temp.set(round(random.uniform(15.0, 35.0), 1))
        self.var_event.set(random.randint(15, 95))
        self.var_booking.set(round(random.uniform(0.15, 0.95), 2))
        self._sync_labels()
        self._predict()

    def _predict(self):
        if not self.pipeline:
            messagebox.showwarning("模型未載入", "模型尚未訓練或載入完成！")
            return

        input_data = pd.DataFrame([{
            "商圈型態": self.var_dist.get(),
            "星期別": self.var_weekday.get(),
            "是否為國定假日": self.var_holiday.get(),
            "天氣狀況": self.var_weather.get(),
            "最高氣溫_度C": float(self.var_temp.get()),
            "商圈活動熱度": int(self.var_event.get()),
            "當日訂位客席率": float(self.var_booking.get())
        }])

        try:
            pred_rev = self.pipeline.predict(input_data)[0]
            est_diners = int(pred_rev / 550) # 假設平均客單價 550 元

            self.res_tag_var.set(f"預估今日營業額：${pred_rev:,.0f} TWD")
            self.cap_var.set(f"預估來客規模：約 {est_diners:,} 人次 (以客單價 $550 估算)")

            if pred_rev >= 220000:
                advice = (
                    f"【店長排班與備料方針：大客流熱門檔期】\n"
                    f"1. 預估營業額高達 ${pred_rev:,.0f} 元，預期滿席翻桌率達 3 輪以上。\n"
                    "2. 外場排班：正職全體到崗，建議加派兼職 PT 人力 3~4 名支援尖峰傳菜收桌。\n"
                    "3. 內場備料：生鮮肉品與海鮮提前於晨間備妥 120% 安全水位，高湯提早熬煮。\n"
                    "4. 現金準備：櫃檯零錢與發票紙捲備妥雙倍，加速結帳避免動線堵塞。"
                )
            elif pred_rev >= 130000:
                advice = (
                    f"【店長排班與備料方針：標準常態營運】\n"
                    f"1. 預估營業額為 ${pred_rev:,.0f} 元，屬於穩定健康營收水準。\n"
                    "2. 外場排班：維持標準正職與常態兼職配置即可，注意午餐尖峰翻桌節奏。\n"
                    "3. 內場備料：按 100% 常態原物料清單備料，減少易腐壞生鮮當日報廢損耗。\n"
                    "4. 若有現場空桌，可引導過路散客入座。"
                )
            else:
                advice = (
                    f"【店長排班與備料方針：雨天淡季節流控管】\n"
                    f"1. 預估營業額僅約 ${pred_rev:,.0f} 元（受天候或平日淡季影響）。\n"
                    "2. 人力節約：適度調整兼職工時，可安排正職人員進行深度設備清潔與庫存盤點。\n"
                    "3. 備料防呆：生鮮蔬菜減少叫貨 25%，肉品以冷凍解凍動態控制，嚴防剩食報廢。\n"
                    "4. 外送平台連動：開啟 UberEats/Foodpanda 促銷折扣，補足內用客流缺口。"
                )

            self.advice_txt.config(state="normal")
            self.advice_txt.delete("1.0", tk.END)
            self.advice_txt.insert("1.0", advice)
            self.advice_txt.config(state="disabled")
        except Exception as e:
            messagebox.showerror("預測錯誤", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = RestaurantRevenueApp(root)
    root.mainloop()
