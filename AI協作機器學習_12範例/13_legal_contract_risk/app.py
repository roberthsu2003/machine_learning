import json
import os
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd

# 取得當前腳本所在目錄，確保相對路徑載入正確
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "artifacts", "metadata.json")

class ContractRiskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("企業合約法律風險預警系統 (Legal Contract Risk AI)")
        self.root.geometry("780x720")
        self.root.minsize(720, 650)

        # 載入模型與元資料
        self.model = None
        self.metadata = None
        self.load_model_artifacts()

        # 設定風格
        self.setup_styles()
        # 建立版面元件
        self.create_widgets()

    def load_model_artifacts(self):
        """僅載入預先訓練完成的模型與元資料，不重新訓練"""
        if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
            messagebox.showerror(
                "找不到模型檔案",
                f"請先完成 lesson.ipynb 訓練並產生 artifacts/model.joblib 與 metadata.json！\n"
                f"路徑: {MODEL_PATH}"
            )
            return

        try:
            self.model = joblib.load(MODEL_PATH)
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        except Exception as e:
            messagebox.showerror("模型載入失敗", f"載入時發生錯誤：\n{str(e)}")

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # 標題與標籤樣式
        self.style.configure("Header.TLabel", font=("Arial", 16, "bold"), foreground="#1a365d")
        self.style.configure("SubHeader.TLabel", font=("Arial", 10), foreground="#4a5568")
        self.style.configure("FieldLabel.TLabel", font=("Arial", 10, "bold"), foreground="#2d3748")
        self.style.configure("TCheckbutton", font=("Arial", 10))
        self.style.configure("Action.TButton", font=("Arial", 11, "bold"), padding=8)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="16 12 16 16")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. 頂部標題區
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(
            header_frame,
            text="⚖️ 企業合約法律風險預警與審查輔助系統",
            style="Header.TLabel"
        ).pack(anchor="w")

        model_acc = self.metadata.get("test_accuracy", "N/A") if self.metadata else "N/A"
        ttk.Label(
            header_frame,
            text=f"專為企業法務、法遵與合約管理專員設計 ｜ 模型測試準確率：{float(model_acc)*100:.1f}%",
            style="SubHeader.TLabel"
        ).pack(anchor="w", pady=(2, 0))

        # 2. 商務與合作方資訊 (LabelFrame)
        biz_group = ttk.LabelFrame(main_frame, text="1. 合約商務條件與合作方背景", padding="12")
        biz_group.pack(fill=tk.X, pady=(0, 10))

        # 第 1 列：合約類型 & 合約金額
        row1 = ttk.Frame(biz_group)
        row1.pack(fill=tk.X, pady=4)

        ttk.Label(row1, text="合約類型：", style="FieldLabel.TLabel", width=12).pack(side=tk.LEFT)
        types = self.metadata.get("contract_types", ["採購合約", "經銷代理", "軟體委外授權", "保密協議NDA", "勞務承攬"])
        self.var_contract_type = tk.StringVar(value=types[0])
        self.combo_type = ttk.Combobox(row1, textvariable=self.var_contract_type, values=types, state="readonly", width=18)
        self.combo_type.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(row1, text="標的金額 (NT$)：", style="FieldLabel.TLabel", width=15).pack(side=tk.LEFT)
        self.var_amount = tk.StringVar(value="1500000")
        self.entry_amount = ttk.Entry(row1, textvariable=self.var_amount, width=16)
        self.entry_amount.pack(side=tk.LEFT)

        # 第 2 列：付款天數 & 合約頁數
        row2 = ttk.Frame(biz_group)
        row2.pack(fill=tk.X, pady=4)

        ttk.Label(row2, text="付款週期天數：", style="FieldLabel.TLabel", width=12).pack(side=tk.LEFT)
        self.var_payment_days = tk.IntVar(value=60)
        self.combo_payment = ttk.Combobox(row2, textvariable=self.var_payment_days, values=[30, 45, 60, 90, 120], state="readonly", width=18)
        self.combo_payment.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(row2, text="合約頁數厚度：", style="FieldLabel.TLabel", width=15).pack(side=tk.LEFT)
        self.var_pages = tk.IntVar(value=15)
        self.spin_pages = ttk.Spinbox(row2, from_=1, to=100, textvariable=self.var_pages, width=14)
        self.spin_pages.pack(side=tk.LEFT)

        # 第 3 列：廠商年資 & 歷史爭議次數
        row3 = ttk.Frame(biz_group)
        row3.pack(fill=tk.X, pady=4)

        ttk.Label(row3, text="合作廠商年資：", style="FieldLabel.TLabel", width=12).pack(side=tk.LEFT)
        self.var_tenure = tk.IntVar(value=5)
        self.spin_tenure = ttk.Spinbox(row3, from_=1, to=50, textvariable=self.var_tenure, width=18)
        self.spin_tenure.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(row3, text="歷史爭議/違約次數：", style="FieldLabel.TLabel", width=15).pack(side=tk.LEFT)
        self.var_disputes = tk.IntVar(value=0)
        self.spin_disputes = ttk.Spinbox(row3, from_=0, to=10, textvariable=self.var_disputes, width=14)
        self.spin_disputes.pack(side=tk.LEFT)

        # 3. 核心法律條款檢核 (LabelFrame)
        clauses_group = ttk.LabelFrame(main_frame, text="2. 關鍵法律條款快速勾選（法務風險要件）", padding="12")
        clauses_group.pack(fill=tk.X, pady=(0, 10))

        self.var_no_liability_cap = tk.BooleanVar(value=False)
        self.var_foreign_law = tk.BooleanVar(value=False)
        self.var_auto_renew = tk.BooleanVar(value=False)
        self.var_ip_transfer = tk.BooleanVar(value=False)

        c1 = ttk.Checkbutton(
            clauses_group,
            text="⚠️ 合約中「無」損害賠償責任上限條款（責任無限）",
            variable=self.var_no_liability_cap
        )
        c1.pack(anchor="w", pady=2)

        c2 = ttk.Checkbutton(
            clauses_group,
            text="🌐 準據法或約定管轄法院為「境外法院 / 外國仲裁」",
            variable=self.var_foreign_law
        )
        c2.pack(anchor="w", pady=2)

        c3 = ttk.Checkbutton(
            clauses_group,
            text="🔄 包含「自動展期 / 自動續約」條款（未提前書面終止即展期）",
            variable=self.var_auto_renew
        )
        c3.pack(anchor="w", pady=2)

        c4 = ttk.Checkbutton(
            clauses_group,
            text="💡 智慧財產權條款「全數讓與對方或由對方獨佔」",
            variable=self.var_ip_transfer
        )
        c4.pack(anchor="w", pady=2)

        # 4. 預測按鈕
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(4, 10))
        self.btn_predict = ttk.Button(
            btn_frame,
            text="🔍 立即評估合約法律風險 (Predict)",
            style="Action.TButton",
            command=self.on_predict
        )
        self.btn_predict.pack(fill=tk.X)

        # 5. 評估結果顯示區
        result_group = ttk.LabelFrame(main_frame, text="3. AI 評估報告與法務建議", padding="12")
        result_group.pack(fill=tk.BOTH, expand=True)

        self.lbl_result_badge = tk.Label(
            result_group,
            text="請輸入合約條件並點擊上方評估按鈕",
            font=("Arial", 14, "bold"),
            bg="#edf2f7",
            fg="#4a5568",
            padx=12,
            pady=8,
            relief=tk.GROOVE
        )
        self.lbl_result_badge.pack(fill=tk.X, pady=(0, 8))

        # 機率明細
        self.lbl_probs = ttk.Label(
            result_group,
            text="",
            font=("Arial", 10),
            foreground="#2d3748"
        )
        self.lbl_probs.pack(anchor="w", pady=(0, 6))

        # 建議事項文字框
        ttk.Label(result_group, text="法務審查策略與因應建議：", style="FieldLabel.TLabel").pack(anchor="w")
        self.txt_advice = tk.Text(result_group, height=6, font=("Arial", 10), wrap=tk.WORD, relief=tk.SOLID, borderwidth=1)
        self.txt_advice.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    def on_predict(self):
        if self.model is None:
            messagebox.showwarning("模型未就緒", "尚未載入模型，請檢查 artifacts 目錄！")
            return

        # 驗證金額輸入
        try:
            amt = float(self.var_amount.get().replace(",", "").strip())
            if amt < 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("數值錯誤", "請輸入有效的合約金額！")
            return

        # 整理輸入特徵 DataFrame
        # 注意：has_liability_cap 1為有上限(低風險)，0為無上限(高風險)
        has_cap = 0 if self.var_no_liability_cap.get() else 1
        # governing_law_local 1為在台本地，0為境外
        gov_local = 0 if self.var_foreign_law.get() else 1
        has_auto = 1 if self.var_auto_renew.get() else 0
        has_ip = 1 if self.var_ip_transfer.get() else 0

        input_data = pd.DataFrame([{
            "contract_type": self.var_contract_type.get(),
            "contract_amount_twd": amt,
            "payment_term_days": int(self.var_payment_days.get()),
            "has_liability_cap": has_cap,
            "governing_law_local": gov_local,
            "has_auto_renewal": has_auto,
            "has_ip_assignment": has_ip,
            "counterparty_tenure_years": int(self.var_tenure.get()),
            "historical_dispute_count": int(self.var_disputes.get()),
            "contract_pages": int(self.var_pages.get())
        }])

        try:
            pred = self.model.predict(input_data)[0]
            classes = list(self.model.classes_)
            probs = self.model.predict_proba(input_data)[0]
            prob_dict = {cls: prob for cls, prob in zip(classes, probs)}
        except Exception as e:
            messagebox.showerror("預測錯誤", f"模型推論時發生錯誤：\n{str(e)}")
            return

        # 更新 Badge
        if pred == "高風險":
            self.lbl_result_badge.config(
                text=f"🚨 評估結果：高風險合約 (High Risk) ｜ 信心度: {prob_dict['高風險']*100:.1f}%",
                bg="#fff5f5",
                fg="#c53030"
            )
        elif pred == "中風險":
            self.lbl_result_badge.config(
                text=f"⚠️ 評估結果：中風險合約 (Medium Risk) ｜ 信心度: {prob_dict['中風險']*100:.1f}%",
                bg="#fffaf0",
                fg="#dd6b20"
            )
        else:
            self.lbl_result_badge.config(
                text=f"✅ 評估結果：低風險合約 (Low Risk) ｜ 信心度: {prob_dict['低風險']*100:.1f}%",
                bg="#f0fff4",
                fg="#2f855a"
            )

        # 顯示各類別機率
        prob_str = " ｜ ".join([f"{k}: {v*100:.1f}%" for k, v in prob_dict.items()])
        self.lbl_probs.config(text=f"預測機率分佈：{prob_str}")

        # 生成法務建議文字
        advice_list = []
        if has_cap == 0:
            advice_list.append("• 【重大風險】未設損害賠償上限：建議增訂「除故意或重大過失外，雙方損害賠償總額以本合約累計收取金額為上限」。")
        if gov_local == 0:
            advice_list.append("• 【重大風險】非本地管轄法院：涉外訴訟成本龐大，強烈建議爭取約定「以臺灣臺北地方法院為第一審管轄法院」。")
        if has_ip == 1:
            advice_list.append("• 【智慧財產權】讓與條款：請確認是否侵害我方既存核心技術，建議改採「非專屬無償授權」而非全數讓與。")
        if has_auto == 1:
            advice_list.append("• 【合約展期】自動續約提醒：建議行事曆註記於到期前 60 天發送書面不續約通知，避免被迫展期。")
        if int(self.var_payment_days.get()) >= 90:
            advice_list.append("• 【財務金流】付款天數超過 90 天：有資金周轉或倒帳風險，建議爭取分期驗收請款。")
        if int(self.var_disputes.get()) >= 2:
            advice_list.append(f"• 【信用風險】該廠商過去有 {self.var_disputes.get()} 次爭議紀錄：建議要求履約保證金或預付款保函。")

        if pred == "低風險" and not advice_list:
            advice_list.append("• 本合約各項商業與法律條款皆合乎標準常規，可適用快速免審流程或由單位主管直接簽核備查。")
        elif pred == "高風險":
            advice_list.insert(0, "【重要】本案屬高風險件，建請退回請合作方修正上述條款，或由資深法務召開談判協調會。")

        self.txt_advice.delete("1.0", tk.END)
        self.txt_advice.insert(tk.END, "\n".join(advice_list))

def main():
    root = tk.Tk()
    app = ContractRiskApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
