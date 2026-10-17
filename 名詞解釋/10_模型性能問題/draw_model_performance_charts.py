import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import mplfonts
from mplfonts import use_font

# 1. 配置中文字型
for f in glob.glob(os.path.join(os.path.dirname(mplfonts.__file__), "fonts", "*.[ot]tf")):
    font_manager.fontManager.addfont(f)

local_font = os.path.abspath(os.path.join("..", "ChineseFont.ttf"))
if os.path.exists(local_font):
    font_manager.fontManager.addfont(local_font)

use_font("Noto Sans CJK SC")
plt.rcParams["axes.unicode_minus"] = False

# 2. 準備共享數據
np.random.seed(42)
# 真實非線性規律 f(x) = 0.45 * x^2 - 1.55
def ground_truth(x):
    return 0.45 * x**2 - 1.55

# 訓練樣本點 (16 個點)
X_train = np.array([-3.0, -2.5, -2.1, -1.7, -1.1, -0.8, -0.2, 0.15, 0.4, 0.8, 1.1, 1.4, 1.75, 2.05, 2.35, 2.7])
noise = np.array([0.7, 0.8, -0.1, -0.3, -0.8, -0.5, 0.1, -0.2, -0.1, 0.2, -0.6, 0.2, 0.5, 0.4, -0.3, 0.6]) * 0.7
y_train = ground_truth(X_train) + noise

# 密集繪圖點
x_dense = np.linspace(-3.2, 3.2, 300)
y_dense_truth = ground_truth(x_dense)

# 測試樣本點 (未見新數據)
X_test_under = np.array([-2.2, 0.0, 2.2])
y_test_under = ground_truth(X_test_under) + np.array([0.15, -0.1, 0.1])

X_test_over = np.array([-2.5, -0.6, 1.2, 2.7])
y_test_over = ground_truth(X_test_over) + np.array([-0.2, 0.1, 0.15, 0.25])


def draw_card(ax, border_color, bg_color, title, title_color, body_lines, footer_badge=None):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # 繪製圓角卡片背景
    card = FancyBboxPatch(
        (0.01, 0.02), 0.98, 0.96,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        edgecolor=border_color,
        facecolor=bg_color,
        linewidth=2.0,
        zorder=1
    )
    ax.add_patch(card)
    
    # 卡片標題
    ax.text(0.045, 0.84, title, fontsize=15.5, color=title_color, va="center", zorder=2)
    
    # 內文 lines
    for line in body_lines:
        ax.text(0.045, line["y"], line["text"], fontsize=line.get("size", 13), 
                color=line.get("color", "#374151"), va="center", zorder=2)
        
    # 底部結論徽章 (若有，位置明確設定在 y=0.08 到 0.32)
    if footer_badge:
        badge_box = FancyBboxPatch(
            (0.035, 0.08), 0.93, 0.24,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            edgecolor=footer_badge["border"],
            facecolor=footer_badge["bg"],
            linewidth=1.4,
            zorder=2
        )
        ax.add_patch(badge_box)
        ax.text(0.055, 0.20, footer_badge["text"], fontsize=12.2,
                color=footer_badge["color"], va="center", zorder=3)


# ==========================================
# 圖 1: 欠擬合 (Underfitting)
# ==========================================
def generate_underfitting_image(out_path):
    fig = plt.figure(figsize=(20, 11), dpi=130, facecolor="#ffffff")
    fig.suptitle("欠擬合 (Underfitting / 高偏差 High Bias) 原理與特徵", fontsize=23, color="#0f172a", y=0.96)
    
    # 使用不等高比例 height_ratios: 卡片2高度設為 1.25，確保三層資訊極致寬敞絕不重疊
    gs = fig.add_gridspec(nrows=4, ncols=2, width_ratios=[1.15, 0.85], 
                           height_ratios=[0.9, 1.25, 1.05, 0.9],
                           left=0.06, right=0.96, bottom=0.06, top=0.88, 
                           hspace=0.22, wspace=0.14)
    
    # --- 左側圖表 ---
    ax_plot = fig.add_subplot(gs[:, 0])
    ax_plot.set_facecolor("#fafafa")
    ax_plot.grid(True, linestyle="--", color="#cbd5e1", alpha=0.5, zorder=0)
    
    # 1. 真實非線性規律
    ax_plot.plot(x_dense, y_dense_truth, linestyle="--", color="#94a3b8", linewidth=2.8, label="真實非線性規律 (Ground Truth)", zorder=2)
    
    # 2. 欠擬合模型 (線性回歸直線)
    poly1 = np.polyfit(X_train, y_train, deg=1)
    y_pred_under_dense = np.polyval(poly1, x_dense)
    y_pred_under_train = np.polyval(poly1, X_train)
    ax_plot.plot(x_dense, y_pred_under_dense, color="#dc2626", linewidth=3.5, label="過於簡單的線性模型 (欠擬合)", zorder=3)
    
    # 3. 殘差線
    for xi, yi, y_hat in zip(X_train, y_train, y_pred_under_train):
        ax_plot.plot([xi, xi], [yi, y_hat], linestyle=":", color="#f87171", linewidth=1.8, alpha=0.85, zorder=1)
        
    # 4. 訓練樣本點
    ax_plot.scatter(X_train, y_train, s=110, color="#2563eb", edgecolor="#1e3a8a", linewidth=1.5, label="訓練樣本點 (Training Points)", zorder=4)
    
    # 5. 未見測試樣本
    ax_plot.scatter(X_test_under, y_test_under, s=260, marker="*", color="#10b981", edgecolor="#065f46", linewidth=1.5, label="未見測試樣本 (Test Data)", zorder=5)
    
    ax_plot.set_title("非線性數據 vs 僵化線性模型（誤差巨大）", fontsize=15.5, pad=12, color="#334155")
    ax_plot.set_xlabel("特徵 X", fontsize=14, labelpad=8, color="#1e293b")
    ax_plot.set_ylabel("目標 y", fontsize=14, labelpad=8, color="#1e293b")
    ax_plot.set_xlim(-3.4, 3.4)
    ax_plot.set_ylim(-2.3, 3.8)
    ax_plot.tick_params(labelsize=12)
    ax_plot.legend(loc="upper left", bbox_to_anchor=(0.03, 0.97), framealpha=0.92, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=11.5)
    
    for spine in ax_plot.spines.values():
        spine.set_color("#94a3b8")
        spine.set_linewidth(1.2)
        
    # --- 右側 4 個卡片 ---
    # 卡片 1: 核心本質
    ax1 = fig.add_subplot(gs[0, 1])
    draw_card(
        ax1, border_color="#f87171", bg_color="#fef2f2",
        title="【核心本質】高偏差 (High Bias)", title_color="#b91c1c",
        body_lines=[
            {"text": "模型結構過度簡化，存在先入為主的死板假設。", "size": 13, "y": 0.55, "color": "#374151"},
            {"text": "根本「學不會」數據中蘊含的真實非線性規律與模式。", "size": 13, "y": 0.28, "color": "#374151"}
        ]
    )
    
    # 卡片 2: 兩大標誌特徵 (高 1.25，Y 軸精確分佈在 0.84, 0.66, 0.48, 徽章在 0.08~0.32)
    ax2 = fig.add_subplot(gs[1, 1])
    draw_card(
        ax2, border_color="#60a5fa", bg_color="#eff6ff",
        title="【兩大標誌特徵】", title_color="#1d4ed8",
        body_lines=[
            {"text": "• 訓練誤差 (Train Error) 很高", "size": 13.5, "color": "#dc2626", "y": 0.66},
            {"text": "• 測試誤差 (Test Error) 同樣很高", "size": 13.5, "color": "#dc2626", "y": 0.48}
        ],
        footer_badge={
            "border": "#93c5fd", "bg": "#dbeafe",
            "color": "#1e40af", "text": "【結論】模型在「已知」與「未知」數據上表現皆不及格！"
        }
    )
    
    # 卡片 3: 常見成因
    ax3 = fig.add_subplot(gs[2, 1])
    draw_card(
        ax3, border_color="#94a3b8", bg_color="#f8fafc",
        title="【常見成因】", title_color="#334155",
        body_lines=[
            {"text": "• 模型容量不足（如非線性問題硬用線性回歸）", "size": 12.5, "y": 0.60, "color": "#475569"},
            {"text": "• 特徵維度太少，缺乏關鍵辨識資訊", "size": 12.5, "y": 0.39, "color": "#475569"},
            {"text": "• 訓練時間不足，或正則化懲罰過重束縛表達力", "size": 12.5, "y": 0.18, "color": "#475569"}
        ]
    )
    
    # 卡片 4: 直觀生活比喻
    ax4 = fig.add_subplot(gs[3, 1])
    draw_card(
        ax4, border_color="#34d399", bg_color="#f0fdf4",
        title="【直觀生活比喻】", title_color="#047857",
        body_lines=[
            {"text": "只背熟了國小「九九乘法表」，就被送上考場應試大學高等微積分，", "size": 12.8, "y": 0.54, "color": "#065f46"},
            {"text": "連考題的數學符號都讀不懂。", "size": 12.8, "y": 0.26, "color": "#065f46"}
        ]
    )
    
    plt.savefig(out_path, dpi=130, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"成功生成欠擬合圖片: {out_path}")


# ==========================================
# 圖 2: 過度擬合 (Overfitting)
# ==========================================
def generate_overfitting_image(out_path):
    fig = plt.figure(figsize=(20, 11), dpi=130, facecolor="#ffffff")
    fig.suptitle("過度擬合 (Overfitting / 高變異數 High Variance) 原理與特徵", fontsize=23, color="#0f172a", y=0.96)
    
    gs = fig.add_gridspec(nrows=4, ncols=2, width_ratios=[1.15, 0.85], 
                           height_ratios=[0.9, 1.25, 1.05, 0.9],
                           left=0.06, right=0.96, bottom=0.06, top=0.88, 
                           hspace=0.22, wspace=0.14)
    
    # --- 左側圖表 ---
    ax_plot = fig.add_subplot(gs[:, 0])
    ax_plot.set_facecolor("#fafafa")
    ax_plot.grid(True, linestyle="--", color="#cbd5e1", alpha=0.5, zorder=0)
    
    # 1. 真實規律
    ax_plot.plot(x_dense, y_dense_truth, linestyle="--", color="#94a3b8", linewidth=2.8, label="真實規律 (Ground Truth)", zorder=2)
    
    # 2. 過擬合模型
    poly_over = np.polyfit(X_train, y_train, deg=11)
    y_pred_over_dense = np.polyval(poly_over, x_dense)
    ax_plot.plot(x_dense, y_pred_over_dense, color="#9333ea", linewidth=3.5, label="過度複雜的高階模型 (過擬合)", zorder=3)
    
    # 3. 訓練樣本點
    ax_plot.scatter(X_train, y_train, s=110, color="#2563eb", edgecolor="#1e3a8a", linewidth=1.5, label="訓練樣本點 (含隨機雜訊)", zorder=4)
    
    # 4. 新測試樣本
    ax_plot.scatter(X_test_over, y_test_over, s=260, marker="*", color="#10b981", edgecolor="#065f46", linewidth=1.5, label="新測試樣本 (Test Data)", zorder=5)
    
    # 標註提示
    ax_plot.annotate("死記雜訊導致\n兩端劇烈翹曲！", xy=(-3.0, 3.8), xytext=(-2.7, 4.5),
                    arrowprops=dict(facecolor="#9333ea", shrink=0.08, width=1.5, headwidth=6),
                    fontsize=11.5, color="#7e22ce")
    
    ax_plot.set_title("死記硬背雜訊 vs 泛化能力崩潰", fontsize=15.5, pad=12, color="#334155")
    ax_plot.set_xlabel("特徵 X", fontsize=14, labelpad=8, color="#1e293b")
    ax_plot.set_ylabel("目標 y", fontsize=14, labelpad=8, color="#1e293b")
    ax_plot.set_xlim(-3.4, 3.4)
    ax_plot.set_ylim(-3.3, 5.3)
    ax_plot.tick_params(labelsize=12)
    ax_plot.legend(loc="upper center", bbox_to_anchor=(0.50, 0.98), framealpha=0.92, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=11.5)
    
    for spine in ax_plot.spines.values():
        spine.set_color("#94a3b8")
        spine.set_linewidth(1.2)
        
    # --- 右側 4 個卡片 ---
    # 卡片 1: 核心本質
    ax1 = fig.add_subplot(gs[0, 1])
    draw_card(
        ax1, border_color="#c084fc", bg_color="#faf5ff",
        title="【核心本質】高變異數 (High Variance)", title_color="#7e22ce",
        body_lines=[
            {"text": "模型容量過大，過分敏感於訓練集中的細微雜訊。", "size": 13, "y": 0.55, "color": "#374151"},
            {"text": "連偶發的數據缺陷與噪點都全盤硬背吸收，喪失泛化能力。", "size": 13, "y": 0.28, "color": "#374151"}
        ]
    )
    
    # 卡片 2: 兩大標誌特徵 (含泛化鴻溝徽章，絕對不重疊)
    ax2 = fig.add_subplot(gs[1, 1])
    draw_card(
        ax2, border_color="#60a5fa", bg_color="#eff6ff",
        title="【兩大標誌特徵】", title_color="#1d4ed8",
        body_lines=[
            {"text": "• 訓練誤差 (Train Error) 極低 (甚至為 0)", "size": 13.5, "color": "#16a34a", "y": 0.66},
            {"text": "• 測試誤差 (Test Error) 嚴重飆高", "size": 13.5, "color": "#dc2626", "y": 0.48}
        ],
        footer_badge={
            "border": "#fca5a5", "bg": "#fee2e2",
            "color": "#b91c1c", "text": "【結論】兩者形成巨大「泛化鴻溝 (Generalization Gap)」！"
        }
    )
    
    # 卡片 3: 常見成因
    ax3 = fig.add_subplot(gs[2, 1])
    draw_card(
        ax3, border_color="#94a3b8", bg_color="#f8fafc",
        title="【常見成因】", title_color="#334155",
        body_lines=[
            {"text": "• 模型過於複雜（參數龐大、樹深無限制）", "size": 12.5, "y": 0.60, "color": "#475569"},
            {"text": "• 訓練數據樣本量太少，無法覆蓋真實分佈", "size": 12.5, "y": 0.39, "color": "#475569"},
            {"text": "• 缺乏正則化約束或迭代訓練次數過多", "size": 12.5, "y": 0.18, "color": "#475569"}
        ]
    )
    
    # 卡片 4: 直觀生活比喻
    ax4 = fig.add_subplot(gs[3, 1])
    draw_card(
        ax4, border_color="#fb923c", bg_color="#fff7ed",
        title="【直觀生活比喻】", title_color="#c2410c",
        body_lines=[
            {"text": "死記硬背考古題的「題號與標點符號」，", "size": 13, "y": 0.54, "color": "#9a3412"},
            {"text": "大考題目稍微抽換數值，立刻交白卷崩潰。", "size": 13, "y": 0.26, "color": "#9a3412"}
        ]
    )
    
    plt.savefig(out_path, dpi=130, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"成功生成過度擬合圖片: {out_path}")


if __name__ == "__main__":
    generate_underfitting_image("images/01_underfitting.png")
    generate_overfitting_image("images/02_overfitting.png")
    
    docs_img_dir = os.path.abspath("../../docs/glossary/10-model-performance/assets/images")
    if os.path.exists(docs_img_dir):
        generate_underfitting_image(os.path.join(docs_img_dir, "01_underfitting.png"))
        generate_overfitting_image(os.path.join(docs_img_dir, "02_overfitting.png"))
