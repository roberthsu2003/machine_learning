#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
營運成長策略 - 行銷活動轉化響應與數據漂移監控系統 (Tkinter 桌面 GUI 實戰版)
------------------------------------------------------------------
功能特點：
1. 完整機器學習全流程：自動載入跨季度會員促銷響應數據、特徵預處理 Pipeline、模型預測與漂移監控。
2. 內建實務營運情境：提供 3 組具代表性的促銷響應情境（高忠誠高折價、一般探索新客、疲乏低響應）一鍵試算。
3. 互動式介面：推播渠道下拉選單，與會員年齡、客單價、App 開啟次數、折價深度滑桿連動。
4. 即時預測與 MLOps 漂移處遇：預估促銷購買機率，並提示模型生命週期監控與再訓練方針。
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
        os.path.join(os.path.dirname(__file__), "campaign_drift.csv"),
        os.path.join(os.getcwd(), "campaign_drift.csv"),
        os.path.join(os.getcwd(), "AI協作機器學習_職場實戰篇", "12_營運_促銷活動成效監控", "campaign_drift.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class CampaignDriftApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 營運成長處 - 促銷活動轉化響應與數據漂移監控系統")
        self.root.geometry("880x820")
        self.root.minsize(820, 760)

        self.pipeline = None
        self.metrics = {}

        self._build_ui()
        self._train_model()

    def _train_model(self):
        data_path = get_data_path()
        if not data_path:
            self.status_var.set("❌ 錯誤：找不到 campaign_drift.csv 資料集！請確認檔案位置。")
            messagebox.showerror("檔案遺失", "找不到 campaign_drift.csv 資料集，請確保其位於當前專案資料夾！")
            return

        try:
            df = pd.read_csv(data_path)
            train_df = df[df["促銷檔期季度"].isin(["2025Q1", "2025Q2", "2025Q3"])]
            test_df = df[df["促銷檔期季度"] == "2025Q4"]

            X_train = train_df.drop(columns=["促銷檔期季度", "是否響應購買"])
            y_train = (train_df["是否響應購買"] == "是").astype(int)

            X_test = test_df.drop(columns=["促銷檔期季度", "是否響應購買"])
            y_test = (test_df["是否響應購買"] == "是").astype(int)

            num_cols = ["會員年齡", "歷史平均客單價_TWD", "近30天App開啟次數", "促銷折價深度"]
            cat_cols = ["推播渠道"]

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
            self.status_var.set(f"✅ 模型訓練就緒 ｜ Q4 漂移評估準確率：{acc*100:.1f}% ｜ ROC-AUC：{auc:.3f} ｜ 樣本規模：{len(df):,} 筆")
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
            text="📈 營運策略與數位增長處 — 促銷響應預測與模型生命週期監控系統",
            font=("Helvetica", 14, "bold"),
            foreground="#0e7490"
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="評估會員對促銷活動的購買意願，並監控市場受眾行為漂移 (Drift)，維持模型生產環境長期精準度",
            font=("Helvetica", 9),
            foreground="#475569"
        ).pack(anchor="w", pady=(2, 0))

        preset_frame = ttk.LabelFrame(main_frame, text="⚡ 快速會員響應情境 (點擊一鍵帶入)", padding="8")
        preset_frame.pack(fill=tk.X, pady=(0, 10))

        btn_box = ttk.Frame(preset_frame)
        btn_box.pack(fill=tk.X)

        ttk.Button(btn_box, text="🔥 情境 A：高轉化黃金客 (高頻開啟/客單價高/7折深度)", command=self._load_preset_high).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="⚖️ 情境 B：一般中階會員 (中度開啟/常規促銷)", command=self._load_preset_medium).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="❄️ 情境 C：優惠疲乏沉睡客 (鮮少開App/淺折價券)", command=self._load_preset_low).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🎲 隨機抽樣會員", command=self._load_random_case).pack(side=tk.LEFT, padx=3)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        input_card = ttk.LabelFrame(content_frame, text="📋 會員屬性與促銷參數設定", padding="12")
        input_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.var_age = tk.IntVar(value=35)
        self.var_spend = tk.IntVar(value=2800)
        self.var_opens = tk.IntVar(value=18)
        self.var_discount = tk.DoubleVar(value=0.30)
        self.var_channel = tk.StringVar(value="LINE官方帳號")

        r = 0
        ttk.Label(input_card, text="會員年齡：").grid(row=r, column=0, sticky="w", pady=4)
        f_age = ttk.Frame(input_card)
        f_age.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_age, from_=18, to=65, variable=self.var_age, command=lambda v: self.lbl_age.config(text=f"{int(float(v))} 歲")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_age = ttk.Label(f_age, text="35 歲", width=8, anchor="e")
        self.lbl_age.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="歷史平均客單價 (TWD)：").grid(row=r, column=0, sticky="w", pady=4)
        f_spd = ttk.Frame(input_card)
        f_spd.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_spd, from_=500, to=8000, variable=self.var_spend, command=lambda v: self.lbl_spd.config(text=f"${int(float(v)):,}")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_spd = ttk.Label(f_spd, text="$2,800", width=10, anchor="e")
        self.lbl_spd.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="近30天 App 開啟次數：").grid(row=r, column=0, sticky="w", pady=4)
        f_opn = ttk.Frame(input_card)
        f_opn.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_opn, from_=0, to=40, variable=self.var_opens, command=lambda v: self.lbl_opn.config(text=f"{int(float(v))} 次")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_opn = ttk.Label(f_opn, text="18 次", width=8, anchor="e")
        self.lbl_opn.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="促銷折價深度：").grid(row=r, column=0, sticky="w", pady=4)
        f_dis = ttk.Frame(input_card)
        f_dis.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_dis, from_=0.10, to=0.50, variable=self.var_discount, command=lambda v: self.lbl_dis.config(text=f"{int(float(v)*100)}% 折扣")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_dis = ttk.Label(f_dis, text="30% 折扣", width=10, anchor="e")
        self.lbl_dis.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="訊息推播渠道：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_channel, values=["LINE官方帳號", "App推播通知", "簡訊SMS", "EDM電子報"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        btn_pred = tk.Button(
            input_card,
            text="🚀 啟動模型評估促銷購買響應率",
            bg="#0e7490",
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief="raised",
            cursor="hand2",
            pady=8,
            command=self._predict
        )
        btn_pred.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(20, 0))

        result_card = ttk.LabelFrame(content_frame, text="📊 轉化機率預測與數據漂移監控", padding="12")
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

        self.prob_var = tk.StringVar(value="預估購買響應機率：-- %")
        self.prob_lbl = tk.Label(result_card, textvariable=self.prob_var, font=("Helvetica", 13, "bold"), fg="#1e293b")
        self.prob_lbl.pack(anchor="w", pady=(0, 4))

        self.prob_bar = ttk.Progressbar(result_card, orient="horizontal", mode="determinate", length=200)
        self.prob_bar.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(result_card, text="💡 行銷推播決策與資料漂移檢核：", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.advice_txt = tk.Text(result_card, height=13, font=("Helvetica", 9), wrap="word", bg="#f8fafc", relief="solid", borderwidth=1)
        self.advice_txt.pack(fill=tk.BOTH, expand=True)
        self.advice_txt.insert("1.0", "調整左側條件或選擇上方情境，系統將預估活動購買機率，並展示如何監控 Q4 數據分佈漂移與模型再訓練機制。")
        self.advice_txt.config(state="disabled")

        self.status_var = tk.StringVar(value="正在初始化機器學習環境與資料集...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w", padding="4")
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def _sync_labels(self):
        self.lbl_age.config(text=f"{int(self.var_age.get())} 歲")
        self.lbl_spd.config(text=f"${int(self.var_spend.get()):,}")
        self.lbl_opn.config(text=f"{int(self.var_opens.get())} 次")
        self.lbl_dis.config(text=f"{int(self.var_discount.get()*100)}% 折扣")

    def _load_preset_high(self):
        self.var_age.set(34)
        self.var_spend.set(3500)
        self.var_opens.set(22)
        self.var_discount.set(0.30)
        self.var_channel.set("LINE官方帳號")
        self._sync_labels()
        self._predict()

    def _load_preset_medium(self):
        self.var_age.set(42)
        self.var_spend.set(2000)
        self.var_opens.set(10)
        self.var_discount.set(0.20)
        self.var_channel.set("App推播通知")
        self._sync_labels()
        self._predict()

    def _load_preset_low(self):
        self.var_age.set(26)
        self.var_spend.set(800)
        self.var_opens.set(2)
        self.var_discount.set(0.10)
        self.var_channel.set("EDM電子報")
        self._sync_labels()
        self._predict()

    def _load_random_case(self):
        self.var_age.set(random.randint(20, 60))
        self.var_spend.set(random.randint(600, 6000))
        self.var_opens.set(random.randint(0, 35))
        self.var_discount.set(random.choice([0.10, 0.15, 0.20, 0.30, 0.40]))
        self.var_channel.set(random.choice(["LINE官方帳號", "App推播通知", "簡訊SMS", "EDM電子報"]))
        self._sync_labels()
        self._predict()

    def _predict(self):
        if not self.pipeline:
            messagebox.showwarning("模型未載入", "模型尚未訓練或載入完成！")
            return

        input_data = pd.DataFrame([{
            "會員年齡": int(self.var_age.get()),
            "歷史平均客單價_TWD": int(self.var_spend.get()),
            "近30天App開啟次數": int(self.var_opens.get()),
            "促銷折價深度": float(self.var_discount.get()),
            "推播渠道": self.var_channel.get()
        }])

        try:
            prob = self.pipeline.predict_proba(input_data)[0][1]
            pct = prob * 100
            self.prob_bar["value"] = pct
            self.prob_var.set(f"預估促銷響應購買機率：{pct:.1f} %")

            if prob >= 0.55:
                self.res_tag_var.set("🔥 高轉化潛力會員 (建議優先推播)")
                self.res_tag_lbl.config(fg="#047857", bg="#d1fae5")
                advice = (
                    "【營運推播建議：精準命中受眾】\n"
                    "1. 響應購買意願極高，建議列入第一批優先發送名單。\n"
                    "2. 由於客單價與活躍度俱佳，可搭配限時 48 小時專屬結帳優惠碼加速轉化。\n"
                    "3. 監控指標：追蹤推播發送後 6 小時內的點擊與加入購物車轉化率。"
                )
            elif prob >= 0.30:
                self.res_tag_var.set("⚖️ 中度響應可能 (建議結合 App 推播)")
                self.res_tag_lbl.config(fg="#b45309", bg="#fef3c7")
                advice = (
                    "【營運推播建議：溫和觸達】\n"
                    "1. 建議使用免費的 App 推播或官網首頁橫幅通知，避免昂貴的簡訊成本。\n"
                    "2. 促銷文案應強調商品使用價值與熱銷排行榜，非單純削價競爭。\n"
                    "3. 設置頻次控管（Frequency Capping），每週不超過 2 次促銷打擾。"
                )
            else:
                self.res_tag_var.set("❄️ 低響應沉睡名單 (建議排除推播節省成本)")
                self.res_tag_lbl.config(fg="#b91c1c", bg="#fee2e2")
                advice = (
                    "【營運推播建議：防疲乏保護機制】\n"
                    "1. 此類受眾對常規折價券反應冷淡，強行發送容易造成 LINE 封鎖或取消訂閱。\n"
                    "2. 建議暫時移出推播受眾包，保留至雙11年度重大檔期再行喚醒。\n"
                    "3. MLOps 漂移警訊：若整個季度的低響應比例異常上升，代表發生了促銷疲乏的概念漂移，應重新檢視促銷吸引力並排程重訓模型！"
                )

            self.advice_txt.config(state="normal")
            self.advice_txt.delete("1.0", tk.END)
            self.advice_txt.insert("1.0", advice)
            self.advice_txt.config(state="disabled")
        except Exception as e:
            messagebox.showerror("預測錯誤", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = CampaignDriftApp(root)
    root.mainloop()
