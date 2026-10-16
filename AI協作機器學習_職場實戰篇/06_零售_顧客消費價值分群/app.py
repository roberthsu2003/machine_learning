#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
零售會員營運 - 顧客消費價值 RFM 智能分群系統 (Tkinter 桌面 GUI 實戰版)
------------------------------------------------------------------
功能特點：
1. 完整機器學習全流程：自動載入會員 RFM 歷史數據、特徵標準化、K-Means 非監督分群與即時分群推論。
2. 內建實務會員情境：提供 4 組具代表性的顧客畫像（頂級 VIP、沉睡客、潛力客、新會員）一鍵載入。
3. 互動式介面：最近消費天數 (R)、消費頻次 (F)、累計金額 (M) 滑桿即時調整試算。
4. 即時預測與分眾行銷處遇：辨識會員分群標籤，並產出專屬 EDM、LINE 優惠券與會員權益經營方針。
"""

import os
import random
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def get_data_path():
    candidates = [
        os.path.join(os.path.dirname(__file__), "customers_rfm.csv"),
        os.path.join(os.getcwd(), "customers_rfm.csv"),
        os.path.join(os.getcwd(), "AI協作機器學習_職場實戰篇", "06_零售_顧客消費價值分群", "customers_rfm.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class CustomerRFMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🏷️ 零售電商會員營運 - RFM 顧客消費價值分群系統 (非監督學習)")
        self.root.geometry("880x820")
        self.root.minsize(820, 760)

        self.pipeline = None
        self.metrics = {}

        self._build_ui()
        self._train_model()

    def _train_model(self):
        data_path = get_data_path()
        if not data_path:
            self.status_var.set("❌ 錯誤：找不到 customers_rfm.csv 資料集！請確認檔案位置。")
            messagebox.showerror("檔案遺失", "找不到 customers_rfm.csv 資料集，請確保其位於當前專案資料夾！")
            return

        try:
            df = pd.read_csv(data_path)
            rfm_cols = ["最近一次消費天數_Recency", "消費頻率次數_Frequency", "累計消費金額_Monetary_TWD"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[rfm_cols])

            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)

            sil = silhouette_score(X_scaled, clusters)

            self.pipeline = {
                "scaler": scaler,
                "kmeans": kmeans,
                "cluster_centers": kmeans.cluster_centers_,
                "features": rfm_cols
            }

            model_save_path = os.path.join(os.path.dirname(data_path), "model.joblib")
            try:
                joblib.dump(self.pipeline, model_save_path)
            except Exception:
                pass

            self.metrics = {"silhouette": sil, "sample_size": len(df)}
            self.status_var.set(f"✅ K-Means 模型分群完成 ｜ 4 大顧客分群 ｜ 輪廓係數：{sil:.3f} ｜ 樣本規模：{len(df):,} 位會員")
        except Exception as e:
            self.status_var.set(f"❌ 模型分群失敗：{str(e)}")
            messagebox.showerror("訓練錯誤", f"分群計算過程發生異常：\n{str(e)}")

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(main_frame)
        header.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            header,
            text="🛍️ 零售與會員行銷團隊 — RFM 顧客價值自動分群系統 (非監督集群篇)",
            font=("Helvetica", 14, "bold"),
            foreground="#047857"
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="演算法自動尋找相似消費行為，無須人工定義標籤，即時將顧客劃分至最佳營運分群",
            font=("Helvetica", 9),
            foreground="#475569"
        ).pack(anchor="w", pady=(2, 0))

        preset_frame = ttk.LabelFrame(main_frame, text="⚡ 快速顧客畫像情境 (點擊一鍵帶入)", padding="8")
        preset_frame.pack(fill=tk.X, pady=(0, 10))

        btn_box = ttk.Frame(preset_frame)
        btn_box.pack(fill=tk.X)

        ttk.Button(btn_box, text="👑 情境 A：頂級 VIP 金牛客 (最近剛買/次數多/高消費)", command=self._load_preset_vip).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="💤 情境 B：沉睡流失風險客 (超過半年未回購)", command=self._load_preset_sleeping).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🚀 情境 C：潛力升級新秀客 (近期活躍回購/累積中)", command=self._load_preset_potential).pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)
        ttk.Button(btn_box, text="🎲 隨機抽樣會員", command=self._load_random_case).pack(side=tk.LEFT, padx=3)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        input_card = ttk.LabelFrame(content_frame, text="📊 會員 RFM 消費特徵輸入", padding="12")
        input_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.var_recency = tk.IntVar(value=14)
        self.var_frequency = tk.IntVar(value=18)
        self.var_monetary = tk.IntVar(value=38000)
        self.var_cat = tk.StringVar(value="美妝保養")

        r = 0
        ttk.Label(input_card, text="最近一次消費天數 (Recency, 天)：").grid(row=r, column=0, sticky="w", pady=4)
        f_rec = ttk.Frame(input_card)
        f_rec.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_rec, from_=1, to=365, variable=self.var_recency, command=lambda v: self.lbl_rec.config(text=f"{int(float(v))} 天前")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_rec = ttk.Label(f_rec, text="14 天前", width=10, anchor="e")
        self.lbl_rec.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="歷史消費頻次 (Frequency, 次)：").grid(row=r, column=0, sticky="w", pady=4)
        f_frq = ttk.Frame(input_card)
        f_frq.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_frq, from_=1, to=40, variable=self.var_frequency, command=lambda v: self.lbl_frq.config(text=f"{int(float(v))} 次")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_frq = ttk.Label(f_frq, text="18 次", width=8, anchor="e")
        self.lbl_frq.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="年度累計消費 (Monetary, TWD)：").grid(row=r, column=0, sticky="w", pady=4)
        f_mon = ttk.Frame(input_card)
        f_mon.grid(row=r, column=1, sticky="ew", pady=4)
        ttk.Scale(f_mon, from_=300, to=80000, variable=self.var_monetary, command=lambda v: self.lbl_mon.config(text=f"${int(float(v)):,}")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_mon = ttk.Label(f_mon, text="$38,000", width=10, anchor="e")
        self.lbl_mon.pack(side=tk.RIGHT)

        r += 1
        ttk.Label(input_card, text="主力偏好消費品類：").grid(row=r, column=0, sticky="w", pady=4)
        ttk.Combobox(input_card, textvariable=self.var_cat, values=["美妝保養", "生鮮食品", "母嬰用品", "服飾配件", "生活小家電"], state="readonly", width=18).grid(row=r, column=1, sticky="ew", pady=4)

        r += 1
        btn_pred = tk.Button(
            input_card,
            text="🚀 啟動 K-Means 演算法判定顧客分群",
            bg="#047857",
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief="raised",
            cursor="hand2",
            pady=8,
            command=self._predict
        )
        btn_pred.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(20, 0))

        result_card = ttk.LabelFrame(content_frame, text="👥 顧客分群定位與會員經營策略", padding="12")
        result_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        self.res_tag_var = tk.StringVar(value="尚未執行分群")
        self.res_tag_lbl = tk.Label(
            result_card,
            textvariable=self.res_tag_var,
            font=("Helvetica", 16, "bold"),
            fg="#047857",
            bg="#ecfdf5",
            pady=12,
            relief="groove"
        )
        self.res_tag_lbl.pack(fill=tk.X, pady=(0, 10))

        self.profile_var = tk.StringVar(value="顧客畫像定位：--")
        self.profile_lbl = tk.Label(result_card, textvariable=self.profile_var, font=("Helvetica", 12, "bold"), fg="#334155")
        self.profile_lbl.pack(anchor="w", pady=(0, 8))

        ttk.Label(result_card, text="🎯 專屬分眾行銷與權益促銷處遇：", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.advice_txt = tk.Text(result_card, height=13, font=("Helvetica", 9), wrap="word", bg="#f8fafc", relief="solid", borderwidth=1)
        self.advice_txt.pack(fill=tk.BOTH, expand=True)
        self.advice_txt.insert("1.0", "調整左側 RFM 參數或點擊上方情境，演算法將即時計算顧客向量與 4 大群集中心的歐氏距離，指派最合適的營運策略。")
        self.advice_txt.config(state="disabled")

        self.status_var = tk.StringVar(value="正在初始化機器學習環境與資料集...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w", padding="4")
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def _sync_labels(self):
        self.lbl_rec.config(text=f"{int(self.var_recency.get())} 天前")
        self.lbl_frq.config(text=f"{int(self.var_frequency.get())} 次")
        self.lbl_mon.config(text=f"${int(self.var_monetary.get()):,}")

    def _load_preset_vip(self):
        self.var_recency.set(7)
        self.var_frequency.set(22)
        self.var_monetary.set(52000)
        self.var_cat.set("美妝保養")
        self._sync_labels()
        self._predict()

    def _load_preset_sleeping(self):
        self.var_recency.set(240)
        self.var_frequency.set(2)
        self.var_monetary.set(1800)
        self.var_cat.set("生活小家電")
        self._sync_labels()
        self._predict()

    def _load_preset_potential(self):
        self.var_recency.set(18)
        self.var_frequency.set(8)
        self.var_monetary.set(13500)
        self.var_cat.set("生鮮食品")
        self._sync_labels()
        self._predict()

    def _load_random_case(self):
        self.var_recency.set(random.randint(3, 300))
        self.var_frequency.set(random.randint(1, 28))
        self.var_monetary.set(random.randint(800, 60000))
        self.var_cat.set(random.choice(["美妝保養", "生鮮食品", "母嬰用品", "服飾配件", "生活小家電"]))
        self._sync_labels()
        self._predict()

    def _predict(self):
        if not self.pipeline:
            messagebox.showwarning("模型未載入", "模型尚未訓練或載入完成！")
            return

        r = int(self.var_recency.get())
        f = int(self.var_frequency.get())
        m = int(self.var_monetary.get())

        try:
            scaler = self.pipeline["scaler"]
            kmeans = self.pipeline["kmeans"]

            input_scaled = scaler.transform([[r, f, m]])
            cluster_id = kmeans.predict(input_scaled)[0]

            # 依據 RFM 實務規則賦予直觀業務標籤
            if m >= 25000 and r <= 60:
                tag = "👑 頂級 VIP 尊榮客群 (高消費/高回購)"
                tag_color = "#047857"
                advice = (
                    "【會員營運處遇建議：VIP 維繫】\n"
                    "1. 專屬福利：配發專屬 1 對 1 私人顧問，享有新品優先預購權與免運特權。\n"
                    "2. 尊榮禮遇：生日當月寄送專屬訂製禮品，強化品牌情感連結。\n"
                    "3. 切忌過度以小折價券打擾，重點在提供無可取代的尊榮服務體驗。"
                )
            elif r >= 150:
                tag = "💤 沉睡瀕危流失客群 (超過5個月未光顧)"
                tag_color = "#b91c1c"
                advice = (
                    "【會員營運處遇建議：緊急喚醒】\n"
                    "1. 喚醒誘因：透過 LINE 官方帳號發送『好久不見！為您保留 $200 專屬回購金』。\n"
                    "2. 滿意度健檢：主動詢問是否有退換貨或使用不滿意經驗，適時化解怨氣。\n"
                    "3. 若連續推播 3 次仍未開啟，降頻發送避免被封鎖帳號。"
                )
            elif f >= 5 and r <= 45:
                tag = "🚀 活躍潛力成長客群 (回購動能強勁)"
                tag_color = "#2563eb"
                advice = (
                    "【會員營運處遇建議：價值升級】\n"
                    "1. 促成滿額：推薦其偏好品類之組合包（Bundle），引導單筆消費突破升級門檻。\n"
                    "2. 忠誠任務：推出『滿 3 次抽大獎』任務活動，將習慣轉化為長期黏著度。\n"
                    "3. 邀請加入 VIP 體驗群組，給予小特權刺激進一步消費。"
                )
            else:
                tag = "🌱 探索型新進客群 (初期接觸/低頻)"
                tag_color = "#d97706"
                advice = (
                    "【會員營運處遇建議：信任培育】\n"
                    "1. 新客教育：提供清楚的新手入門教學與明星熱銷排行榜推薦。\n"
                    "2. 次購激勵：下單 14 天內提供第二次購買 88 折優惠券，防止變成一次性顧客。\n"
                    "3. 鼓勵綁定 LINE 官方帳號領取開卡見面禮。"
                )

            self.res_tag_var.set(tag)
            self.res_tag_lbl.config(fg=tag_color)
            self.profile_var.set(f"分群 ID: 第 {cluster_id} 群 ｜ 最近天數：{r}天 ｜ 頻次：{f}次 ｜ 累計金額：${m:,}")

            self.advice_txt.config(state="normal")
            self.advice_txt.delete("1.0", tk.END)
            self.advice_txt.insert("1.0", advice)
            self.advice_txt.config(state="disabled")
        except Exception as e:
            messagebox.showerror("分群錯誤", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = CustomerRFMApp(root)
    root.mainloop()
