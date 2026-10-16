#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能客服系統 - 顧客評價自然語言情緒分析與工單派發 (Tkinter 桌面 GUI 實戰版)
------------------------------------------------------------------
功能特點：
1. 完整機器學習全流程：自動載入顧客文本評論、TF-IDF 文本特徵提取 Pipeline、邏輯回歸情緒推論。
2. 內建實務客訴情境：提供 3 組具代表性的顧客評論（極佳滿意好評、嚴重客訴退貨、中性諮詢）一鍵載入。
3. 互動式介面：文字輸入框自由輸入任意中文評價，即時計算正面 / 負面情緒機率。
4. 即時預警與客服工單處遇：預警嚴重客訴並產出公關危機處置與感謝致意回覆範本。
"""

import os
import random
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score


def get_data_path():
    candidates = [
        os.path.join(os.path.dirname(__file__), "reviews.csv"),
        os.path.join(os.getcwd(), "reviews.csv"),
        os.path.join(os.getcwd(), "AI協作機器學習_職場實戰篇", "11_客服_顧客評論情緒分析", "reviews.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class ReviewSentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("💬 智能客服中心 - 顧客評論語意情緒分析系統 (NLP篇)")
        self.root.geometry("880x820")
        self.root.minsize(820, 760)

        self.pipeline = None
        self.metrics = {}

        self._build_ui()
        self._train_model()

    def _train_model(self):
        data_path = get_data_path()
        if not data_path:
            self.status_var.set("❌ 錯誤：找不到 reviews.csv 資料集！請確認檔案位置。")
            messagebox.showerror("檔案遺失", "找不到 reviews.csv 資料集，請確保其位於當前專案資料夾！")
            return

        try:
            df = pd.read_csv(data_path)
            X = df["顧客評論內容"]
            y = (df["情緒標籤"] == "正面").astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            self.pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=500)),
                ("classifier", LogisticRegression(random_state=42))
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
            self.status_var.set(f"✅ NLP 模型訓練就緒 ｜ 測試集準確率：{acc*100:.1f}% ｜ ROC-AUC：{auc:.3f} ｜ 樣本規模：{len(df):,} 筆評論")
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
            text="💬 顧客關係與品質管理處 — 評論情緒自動辨識與客訴分流系統 (NLP篇)",
            font=("Helvetica", 14, "bold"),
            foreground="#6d28d9"
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="自然語言文字特徵自動提取 (TF-IDF)，即時研判評論情緒正負向，並自動產出公關回覆指引",
            font=("Helvetica", 9),
            foreground="#475569"
        ).pack(anchor="w", pady=(2, 0))

        preset_frame = ttk.LabelFrame(main_frame, text="⚡ 快速顧客評價範例 (點擊一鍵帶入)", padding="8")
        preset_frame.pack(fill=tk.X, pady=(0, 10))

        btn_box = ttk.Frame(preset_frame)
        btn_box.pack(fill=tk.X)

        ttk.Button(btn_box, text="🌟 範例 A：極佳滿意 (物流快/包裝好/品質高)", command=self._load_preset_pos).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🚨 範例 B：嚴重客訴 (出貨慢/客服敷衍/破損)", command=self._load_preset_neg).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="⚠️ 範例 C：品質失望 (色差嚴重/廉價材質)", command=self._load_preset_neg2).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🎲 隨機抽樣評論", command=self._load_random_case).pack(side=tk.LEFT, padx=3)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        input_card = ttk.LabelFrame(content_frame, text="📝 顧客評論內容輸入 (可自由打字編輯)", padding="12")
        input_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        ttk.Label(input_card, text="輸入顧客回饋文字（支援繁體中文）：").pack(anchor="w", pady=(0, 4))
        self.comment_txt = tk.Text(input_card, height=12, font=("Helvetica", 10), wrap="word", relief="solid", borderwidth=1)
        self.comment_txt.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.comment_txt.insert("1.0", "物流速度非常快，包裝完整沒有破損，商品品質超棒！")

        btn_pred = tk.Button(
            input_card,
            text="🚀 啟動 NLP 模型分析文字情緒與意圖",
            bg="#6d28d9",
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief="raised",
            cursor="hand2",
            pady=8,
            command=self._predict
        )
        btn_pred.pack(fill=tk.X, pady=(5, 0))

        result_card = ttk.LabelFrame(content_frame, text="📊 情緒分析結果與客服公關回覆指引", padding="12")
        result_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        self.res_tag_var = tk.StringVar(value="尚未執行分析")
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

        self.prob_var = tk.StringVar(value="正面情緒信心度：-- %")
        self.prob_lbl = tk.Label(result_card, textvariable=self.prob_var, font=("Helvetica", 13, "bold"), fg="#1e293b")
        self.prob_lbl.pack(anchor="w", pady=(0, 4))

        self.prob_bar = ttk.Progressbar(result_card, orient="horizontal", mode="determinate", length=200)
        self.prob_bar.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(result_card, text="🎯 客服工單分級與回覆範本處遇：", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.advice_txt = tk.Text(result_card, height=13, font=("Helvetica", 9), wrap="word", bg="#f8fafc", relief="solid", borderwidth=1)
        self.advice_txt.pack(fill=tk.BOTH, expand=True)
        self.advice_txt.insert("1.0", "在左側輸入顧客評論並點擊分析，NLP 演算法將即時解析字詞權重，輸出情緒傾向、工單處理優先級與官方回覆建議範本。")
        self.advice_txt.config(state="disabled")

        self.status_var = tk.StringVar(value="正在初始化機器學習環境與資料集...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w", padding="4")
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def _load_preset_pos(self):
        self.comment_txt.delete("1.0", tk.END)
        self.comment_txt.insert("1.0", "物流速度非常快，包裝完整沒有破損，商品品質超棒！客服回覆也很親切！")
        self._predict()

    def _load_preset_neg(self):
        self.comment_txt.delete("1.0", tk.END)
        self.comment_txt.insert("1.0", "出貨慢到誇張，等了快三個禮拜才到，客服一直官腔敷衍，非常糟糕！")
        self._predict()

    def _load_preset_neg2(self):
        self.comment_txt.delete("1.0", tk.END)
        self.comment_txt.insert("1.0", "收到商品發現外殼有明顯刮痕破損，品質控管太差勁了，千萬別買！")
        self._predict()

    def _load_random_case(self):
        presets = [
            "質感超出預期，CP值超高，已經推薦給辦公室同事了。",
            "包裝爛得像被車碾過，裡面的配件還少給，態度消極不負責！",
            "出貨效率很高，贈品也很精緻，滿意的購物體驗！",
            "跟照片完全不一樣，色差超級嚴重，材質像廉價塑膠，很失望！",
            "才用兩天就故障壞掉，品質完全對不起這個價格，要求立即退款！"
        ]
        self.comment_txt.delete("1.0", tk.END)
        self.comment_txt.insert("1.0", random.choice(presets))
        self._predict()

    def _predict(self):
        if not self.pipeline:
            messagebox.showwarning("模型未載入", "模型尚未訓練或載入完成！")
            return

        text = self.comment_txt.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("輸入為空", "請輸入顧客評論內容！")
            return

        try:
            pos_prob = self.pipeline.predict_proba([text])[0][1]
            pct = pos_prob * 100
            self.prob_bar["value"] = pct
            self.prob_var.set(f"正面情緒信心度：{pct:.1f} % (負面機率：{100-pct:.1f}%)")

            if pos_prob >= 0.5:
                self.res_tag_var.set("😊 正向肯定好評 (滿意體驗)")
                self.res_tag_lbl.config(fg="#15803d", bg="#dcfce7")
                advice = (
                    "【客服處遇建議：好評感謝維繫】\n"
                    "1. 工單等級：一般日常維護（可由客服範本快速自動回覆）。\n"
                    "2. 官方回覆範本：『感謝您的肯定與支持！您的滿意是我們最大的動力，期待未來再次為您服務，祝您生活愉快！』\n"
                    "3. 品牌擴散：邀請顧客將好評同步分享至 Google 商家或個人社群，領取專屬回饋紅利金。"
                )
            else:
                self.res_tag_var.set("😡 負向客訴不滿 (需專人介入)")
                self.res_tag_lbl.config(fg="#b91c1c", bg="#fee2e2")
                advice = (
                    "【客服處遇建議：嚴重客訴緊急排解】\n"
                    "1. 工單等級：🔥 P1 最高優先級（需於 2 小時內由資深客服專員致電）。\n"
                    "2. 處置流程：立即啟動『道歉＋免運換新/退款』專案處理，避免爭議升級至消保官。\n"
                    "3. 官方回覆範本：『非常抱歉造成您的困擾與不佳體驗！我們已第一時間將此案列為最高優先級，主管將於 1 小時內親自與您聯繫協助換貨補償事宜，請您放心！』\n"
                    "4. 品質回饋：將破損或延遲紀錄同步反饋至倉儲物流部門進行防呆檢討。"
                )

            self.advice_txt.config(state="normal")
            self.advice_txt.delete("1.0", tk.END)
            self.advice_txt.insert("1.0", advice)
            self.advice_txt.config(state="disabled")
        except Exception as e:
            messagebox.showerror("分析錯誤", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = ReviewSentimentApp(root)
    root.mainloop()
