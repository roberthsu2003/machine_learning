#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
烘焙生鮮門市 - 每日品項備料出爐與銷量預估系統 (Tkinter 桌面 GUI 實戰版)
------------------------------------------------------------------
功能特點：
1. 完整機器學習全流程：自動載入烘焙門市歷史銷售與備料數據、特徵預處理 Pipeline、隨機森林迴歸模型推論。
2. 內建實務烘焙情境：提供 3 組具代表性的出爐情境（週末招牌可頌大賣、平日生吐司穩定出爐、雨天減量備料）一鍵試算。
3. 互動式介面：品項、天氣、星期下拉選單，與前日銷量、週均銷量滑桿即時連動。
4. 即時預估與減損處遇：精確預估明日建議出爐烘烤顆數，並產出麵團發酵排程與剩食防範方針。
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
        os.path.join(os.path.dirname(__file__), "bakery_demand.csv"),
        os.path.join(os.getcwd(), "bakery_demand.csv"),
        os.path.join(os.getcwd(), "AI協作機器學習_職場實戰篇", "10_門市_烘焙生鮮備料預估", "bakery_demand.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class BakeryDemandApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🥐 烘焙生鮮門市 - 每日出爐備料需求量預估系統")
        self.root.geometry("880x820")
        self.root.minsize(820, 760)

        self.pipeline = None
        self.metrics = {}

        self._build_ui()
        self._train_model()

    def _train_model(self):
        data_path = get_data_path()
        if not data_path:
            self.status_var.set("❌ 錯誤：找不到 bakery_demand.csv 資料集！請確認檔案位置。")
            messagebox.showerror("檔案遺失", "找不到 bakery_demand.csv 資料集，請確保其位於當前專案資料夾！")
            return

        try:
            df = pd.read_csv(data_path)
            X = df.drop(columns=["預估次日備料需求量_個"])
            y = df["預估次日備料需求量_個"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            num_cols = ["最高氣溫_度C", "前一日實際銷量_個", "過去一週平均銷量_個"]
            cat_cols = ["烘焙品項", "星期幾", "天氣狀況", "當日是否商圈特賣"]

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
            self.status_var.set(f"✅ 模型訓練就緒 ｜ 測試集解釋力 R²：{r2:.3f} ｜ 平均絕對誤差 MAE：{mae:.1f} 個 ｜ 樣本規模：{len(df):,} 筆紀錄")
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
            text="🍞 烘焙門市與生鮮熟食主廚 — 次日備料出爐量精準預估系統 (迴歸篇)",
            font=("Helvetica", 14, "bold"),
            foreground="#d97706"
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="輸入前日銷量、週平均、天候與商圈活動，AI 即時精準推算次日烘烤顆數，防範下架報廢損耗",
            font=("Helvetica", 9),
            foreground="#475569"
        ).pack(anchor="w", pady=(2, 0))

        preset_frame = ttk.LabelFrame(main_frame, text="⚡ 快速備料情境 (點擊一鍵帶入)", padding="8")
        preset_frame.pack(fill=tk.X, pady=(0, 10))

        btn_box = ttk.Frame(preset_frame)
        btn_box.pack(fill=tk.X)

        ttk.Button(btn_box, text="🥐 情境 A：週末可頌熱門出爐 (週六/特賣活動/高前銷量)", command=self._load_preset_high).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🍞 情境 B：平日生吐司常態 (週三/多雲/常態均值)", command=self._load_preset_medium).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🌧️ 情境 C：下雨天減量出爐 (週二/大雨冷冽/前日銷量低)", command=self._load_preset_low).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🎲 隨機抽樣品項", command=self._load_random_case).pack(side=tk.LEFT, padx=3)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        input_card = ttk.LabelFrame(content_frame, text="📋 品項屬性與銷量前兆特徵", padding="12")
        input_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.var_item = tk.StringVar(value="招牌法式可頌")
        self.var_day = tk.StringVar(value="星期六")
        self.var_weather = tk.StringVar(value="晴朗多雲")
        self.var_temp = tk.DoubleVar(value=26.0)
        self.var_prev = tk.IntVar(value=190)
        self.var_week_avg = tk.IntVar(value=185)
        self.var_event = tk.StringVar(value="是")

        r = 0
        ttk.Label(input_card, text="烘焙商品品項：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_item, values=["招牌法式可頌", "海鹽奶油生吐司", "波士頓生乳捲", "歐式核桃雜糧麵包", "肉桂捲"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="出爐星期別：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_day, values=["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="氣象天氣狀況：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_weather, values=["晴朗多雲", "晴朗高溫", "局部陣雨", "大雨冷冽"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        ttk.Label(input_card, text="最高氣溫 (度C)：").grid(row=r, column=0, sticky="w", pady=4)
        f_tmp = ttk.Frame(input_card)
        f_tmp.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_tmp, from_=10, to=38, variable=self.var_temp, command=lambda v: self.lbl_tmp.config(text=f"{float(v):.1f} °C")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_tmp = ttk.Label(f_tmp, text="26.0 °C", width=8, anchor="e")
        self.lbl_tmp.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="前一日實際銷量 (個)：").grid(row=r, column=0, sticky="w", pady=4)
        f_pre = ttk.Frame(input_card)
        f_pre.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_pre, from_=20, to=350, variable=self.var_prev, command=lambda v: self.lbl_pre.config(text=f"{int(float(v))} 個")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_pre = ttk.Label(f_pre, text="190 個", width=8, anchor="e")
        self.lbl_pre.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="過去一週平均銷量 (個)：").grid(row=r, column=0, sticky="w", pady=4)
        f_avg = ttk.Frame(input_card)
        f_avg.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_avg, from_=20, to=350, variable=self.var_week_avg, command=lambda v: self.lbl_avg.config(text=f"{int(float(v))} 個")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_avg = ttk.Label(f_avg, text="185 個", width=8, anchor="e")
        self.lbl_avg.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="當日是否商圈大特賣：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_event, values=["是", "否"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        btn_pred = tk.Button(
            input_card,
            text="🚀 啟動迴歸模型估算次日出爐備料需求量",
            bg="#d97706",
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief="raised",
            cursor="hand2",
            pady=8,
            command=self._predict
        )
        btn_pred.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(15, 0))

        result_card = ttk.LabelFrame(content_frame, text="📊 備料需求預估與烘焙排程指引", padding="12")
        result_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        self.res_tag_var = tk.StringVar(value="尚未執行估算")
        self.res_tag_lbl = tk.Label(
            result_card,
            textvariable=self.res_tag_var,
            font=("Helvetica", 18, "bold"),
            fg="#d97706",
            bg="#fef3c7",
            pady=12,
            relief="groove"
        )
        self.res_tag_lbl.pack(fill=tk.X, pady=(0, 10))

        self.batch_var = tk.StringVar(value="建議出爐批次安排：-- 烤盤")
        self.batch_lbl = tk.Label(result_card, textvariable=self.batch_var, font=("Helvetica", 12, "bold"), fg="#334155")
        self.batch_lbl.pack(anchor="w", pady=(0, 8))

        ttk.Label(result_card, text="💡 烘焙師傅晨間備料與減損方針：", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.advice_txt = tk.Text(result_card, height=13, font=("Helvetica", 9), wrap="word", bg="#f8fafc", relief="solid", borderwidth=1)
        self.advice_txt.pack(fill=tk.BOTH, expand=True)
        self.advice_txt.insert("1.0", "調整左側銷量與天候或選擇上方情境，系統將根據歷史銷量與氣候數據，預估出爐顆數並產出烤盤排程與防損方針。")
        self.advice_txt.config(state="disabled")

        self.status_var = tk.StringVar(value="正在初始化機器學習環境與資料集...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w", padding="4")
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def _sync_labels(self):
        self.lbl_tmp.config(text=f"{float(self.var_temp.get()):.1f} °C")
        self.lbl_pre.config(text=f"{int(self.var_prev.get())} 個")
        self.lbl_avg.config(text=f"{int(self.var_week_avg.get())} 個")

    def _load_preset_high(self):
        self.var_item.set("招牌法式可頌")
        self.var_day.set("星期六")
        self.var_weather.set("晴朗多雲")
        self.var_temp.set(27.0)
        self.var_prev.set(210)
        self.var_week_avg.set(195)
        self.var_event.set("是")
        self._sync_labels()
        self._predict()

    def _load_preset_medium(self):
        self.var_item.set("海鹽奶油生吐司")
        self.var_day.set("星期三")
        self.var_weather.set("晴朗多雲")
        self.var_temp.set(24.0)
        self.var_prev.set(120)
        self.var_week_avg.set(118)
        self.var_event.set("否")
        self._sync_labels()
        self._predict()

    def _load_preset_low(self):
        self.var_item.set("波士頓生乳捲")
        self.var_day.set("星期二")
        self.var_weather.set("大雨冷冽")
        self.var_temp.set(15.0)
        self.var_prev.set(45)
        self.var_week_avg.set(50)
        self.var_event.set("否")
        self._sync_labels()
        self._predict()

    def _load_random_case(self):
        self.var_item.set(random.choice(["招牌法式可頌", "海鹽奶油生吐司", "波士頓生乳捲", "歐式核桃雜糧麵包", "肉桂捲"]))
        self.var_day.set(random.choice(["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]))
        self.var_weather.set(random.choice(["晴朗多雲", "晴朗高溫", "局部陣雨", "大雨冷冽"]))
        self.var_temp.set(round(random.uniform(12.0, 35.0), 1))
        self.var_prev.set(random.randint(40, 220))
        self.var_week_avg.set(random.randint(40, 220))
        self.var_event.set(random.choice(["是", "否"]))
        self._sync_labels()
        self._predict()

    def _predict(self):
        if not self.pipeline:
            messagebox.showwarning("模型未載入", "模型尚未訓練或載入完成！")
            return

        input_data = pd.DataFrame([{
            "烘焙品項": self.var_item.get(),
            "星期幾": self.var_day.get(),
            "天氣狀況": self.var_weather.get(),
            "最高氣溫_度C": float(self.var_temp.get()),
            "前一日實際銷量_個": int(self.var_prev.get()),
            "過去一週平均銷量_個": int(self.var_week_avg.get()),
            "當日是否商圈特賣": self.var_event.get()
        }])

        try:
            pred_demand = int(self.pipeline.predict(input_data)[0])
            trays = int(np.ceil(pred_demand / 12)) # 每烤盤 12 個

            self.res_tag_var.set(f"建議次日出爐量：{pred_demand} 個")
            self.batch_var.set(f"標準烤盤排程：約 {trays} 烤盤 (以每盤 12 個計算)")

            advice = (
                f"【主廚備料排程建議：{self.var_item.get()}】\n"
                f"1. 預估需求量為 {pred_demand} 個，建議麵團於今日下午完成基礎發酵並冷藏壓麵。\n"
                f"2. 分批出爐策略：建議晨間 07:30 首批出爐 50%，午後 15:30 補充 35%，傍晚 17:30 補足最後 15%。\n"
                "3. 即期減損防呆：若晚上 20:00 架上剩餘超過 10 個，啟動 LINE 友熟客專屬 75 折晚鳥惜食優惠。\n"
                "4. 麵粉奶油備貨：原物料叫貨量請維持至少 3 天的安全周轉庫存。"
            )

            self.advice_txt.config(state="normal")
            self.advice_txt.delete("1.0", tk.END)
            self.advice_txt.insert("1.0", advice)
            self.advice_txt.config(state="disabled")
        except Exception as e:
            messagebox.showerror("預測錯誤", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = BakeryDemandApp(root)
    root.mainloop()
