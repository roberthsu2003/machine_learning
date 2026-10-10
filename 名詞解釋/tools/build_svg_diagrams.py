"""Redraw suitable teaching diagrams as real SVG (no embedded bitmap).

Run from the repository root: .venv/bin/python 名詞解釋/tools/build_svg_diagrams.py
Original PNGs remain available; plot-heavy figures are deliberately untouched.
"""
from pathlib import Path
from html import escape
import json
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / '名詞解釋'
COLORS = [('#2563eb', '#eff6ff'), ('#16834a', '#f0fdf4'),
          ('#b86a10', '#fffbeb'), ('#8053bc', '#faf5ff')]


def wrap(text, max_units):
    result, line, units = [], '', 0
    tokens = re.findall(r'[A-Za-z0-9_]+(?:[./%～−+-][A-Za-z0-9_]+)*%?|[^A-Za-z0-9_]', text)
    for token in tokens:
        n = sum(2 if unicodedata.east_asian_width(char) in 'WF' else 1 for char in token)
        punctuation = token in '。，、；：！？）］》%'
        if units + n > max_units and line and not punctuation:
            result.append(line)
            line, units = '', 0
        line += token
        units += n
    if line:
        result.append(line)
    return result


class SVG:
    def __init__(self, title, height):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title>',
                      '<desc id="desc">以向量圖形和文字重製的教學圖解。原始 PNG 保留於同一資料夾。</desc>',
                      '<style>text{font-family:"Noto Sans CJK TC","Microsoft JhengHei","PingFang TC",sans-serif;fill:#1e293b} .heading{font-weight:700}</style>',
                      '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="none" stroke="#64748b" stroke-width="1.5"/></marker></defs>',
                      f'<rect width="1200" height="{height}" fill="#fff"/>']
        self.text(600, 43, title, 29, center=True, bold=True)

    def text(self, x, y, value, size=22, center=False, bold=False, color=None):
        attrs = f'font-size="{size}"' + (' text-anchor="middle"' if center else '')
        if bold:
            attrs += ' class="heading"'
        if color:
            attrs += f' style="fill:{color}"'
        self.parts.append(f'<text x="{x}" y="{y}" {attrs}>{escape(str(value))}</text>')

    def rect(self, x, y, w, h, color=0):
        stroke, fill = COLORS[color % len(COLORS)]
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

    def arrow(self, x1, y1, x2, y2):
        self.parts.append(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>')

    def card(self, x, y, w, h, title, lines, color=0):
        self.rect(x, y, w, h, color)
        for i, line in enumerate(wrap(title, int((w - 40) / 13))):
            self.text(x + 20, y + 35 + i * 31, line, 24, bold=True, color=COLORS[color % 4][0])
        yy = y + 46 + len(wrap(title, int((w - 40) / 13))) * 31
        for item in lines:
            for line in wrap(item, int((w - 40) / 11)):
                self.text(x + 20, yy, line)
                yy += 31
            yy += 14
        assert yy - 14 < y + h + 8, (title, yy, y + h)

    def save(self, path):
        self.parts.append('</svg>')
        path.write_text('\n'.join(self.parts) + '\n')


def cards(title, items, columns=2, footer=''):
    w = (1120 - 24 * (columns - 1)) / columns
    row_heights = []
    for start in range(0, len(items), columns):
        row_heights.append(max(110 + 31 * (len(wrap(t, int((w-40)/13))) - 1)
                               + sum(31 * len(wrap(s, int((w-40)/11))) + 14 for s in lines)
                               for t, lines in items[start:start+columns]))
    height = 85 + sum(row_heights) + 24 * len(row_heights) + (55 if footer else 0)
    svg = SVG(title, height)
    y = 85
    for start, h in zip(range(0, len(items), columns), row_heights):
        for col, (t, lines) in enumerate(items[start:start+columns]):
            svg.card(40 + col * (w+24), y, w, h, t, lines, start+col)
        y += h + 24
    if footer:
        svg.text(600, height-23, footer, 20, center=True)
    return svg


def flow(title, items, footer='', columns=3):
    svg = cards(title, items, columns, footer)
    # Arrows occupy the gaps between cards; later rows continue at the left.
    w = (1120 - 24 * (columns - 1)) / columns
    y = 85
    for start in range(0, len(items), columns):
        h = max(110 + 31 * (len(wrap(t, int((w-40)/13))) - 1)
                + sum(31 * len(wrap(s, int((w-40)/11))) + 14 for s in lines)
                for t, lines in items[start:start+columns])
        for col in range(min(columns, len(items)-start)-1):
            x = 40 + col*(w+24) + w
            svg.arrow(x+2, y+h/2, x+22, y+h/2)
        y += h+24
    return svg


specs = {}


def add(ch, filename, svg):
    folder = next(BASE.glob(f'{ch:02d}_*'))
    path = folder / 'images' / (filename+'.svg')
    assert path.with_suffix('.png').exists()
    svg.save(path)
    specs[str(path.with_suffix('.png').relative_to(ROOT))] = str(path.relative_to(ROOT))


add(1, '03_comparison', cards('機器學習的類型：監督式與非監督式', [
    ('監督式學習 Supervised Learning', ['輸入：特徵 X ＋已知標籤 y。', '目標：學出從 X 到 y 的預測規則。', '分類：預測離散類別，例如垃圾郵件／一般郵件。', '回歸：預測連續數值，例如房價、氣溫。', '評估：比較預測 ŷ 與真實標籤 y。']),
    ('非監督式學習 Unsupervised Learning', ['輸入：只有特徵 X，沒有提供標籤 y。', '目標：尋找資料內部的結構與相似性。', '分群：將相似樣本分在一起，例如顧客分群。', '降維：用較少的特徵表示資料，例如 PCA。', '評估：依任務檢查群內相似性或資訊保留程度。'])], footer='補充：半監督式使用少量標籤；強化學習以行動與獎勵回饋學習。'))

# Table: keep the original figure's four samples and two features.
s = SVG('資料的表格結構：特徵矩陣 X 與標籤向量 y', 540)
s.rect(40, 85, 1120, 270, 0)
rows = [['樣本', '面積 X₁（m²）', '房間數 X₂', '售價 y（萬元）'],
        ['樣本 1', '100', '2', '500'], ['樣本 2', '80', '1', '300'],
        ['樣本 3', '120', '3', '600'], ['樣本 4', '90', '2', '420']]
for i, row in enumerate(rows):
    for j, value in enumerate(row):
        s.text(180+j*280, 125+i*48, value, 22, center=True, bold=i==0, color='#16834a' if j==3 else None)
s.card(40, 380, 545, 130, 'X：4 筆樣本 × 2 個特徵', ['每列是一筆樣本，每欄是一個特徵。'], 0)
s.card(615, 380, 545, 130, 'y：對應每筆樣本的答案', ['輸入特徵不包含要預測的售價。'], 1)
add(2, '03_data_structure_example', s)

add(3, '01_train_test_split', cards('訓練集與測試集：學習和驗收分開', [
    ('訓練集 Train：80%（比例示例）', ['用途：讓模型學習規律、調整參數。', '生活比喻：平時作業與練習題。', '可反覆使用：更新權重 w 與偏置 b。', '注意：訓練表現好，不代表能預測新資料。']),
    ('測試集 Test：20%（比例示例）', ['用途：模型定型後，評估未見資料的表現。', '生活比喻：最後的期末考。', '不可用於訓練、挑模型或調超參數。', '注意：不得把測試資訊洩漏進訓練。'])], footer='兩份資料的樣本不重疊；實際比例依資料量與任務決定。'))
add(3, '02_three_way_split_validation', cards('三份資料的分工：Train／Validation／Test', [
    ('1. 訓練集 Train：70%', ['學習模型參數。', '例如：權重、偏置。', '用於擬合與參數更新。']),
    ('2. 驗證集 Validation：15%', ['挑選模型與超參數。', '例如：樹深度、學習率。', '監控驗證損失與早停。']),
    ('3. 測試集 Test：15%', ['模型定型後最終驗收。', '檢查未見資料的泛化。', '不參與訓練或調參。'])], columns=3, footer='70／15／15 是比例示例，不是所有專案的固定規則。'))
add(3, '03_split_workflow_and_overfitting', flow('先切分資料，再訓練與評估', [
    ('1. 切分與預處理', ['先分 Train／Val／Test。', '預處理只在 Train fit。', '其餘資料只 transform。']),
    ('2. 訓練與挑選', ['Train：學模型參數。', 'Validation：挑設定。', 'Test：最後才使用。']),
    ('3. 診斷與改善', ['訓練／驗證都高：欠擬合。', '訓練低、驗證高：過度擬合。', '兩者都低：較理想的擬合。'])], footer='評估時同時看誤差高度與差距；避免用 Test 反覆調整模型。'))

add(4, '01_feature_selection', cards('特徵選擇：三種方法的差別', [
    ('過濾法 Filter', ['不先訓練模型，以統計指標評分。', '方法：變異數、相關係數、卡方、互資訊。', '優點：快，適合大量特徵初篩。', '限制：容易忽略特徵間的交互作用。']),
    ('包裝法 Wrapper', ['反覆訓練模型，評估不同特徵組合。', '方法：RFE、前向搜尋、後向淘汰。', '優點：針對模型評估組合效果。', '限制：計算成本高，也須防止過度擬合。']),
    ('嵌入法 Embedded', ['在模型訓練過程中選擇特徵。', '方法：L1／Lasso、樹模型特徵重要性。', '優點：將訓練與特徵選擇結合。', '限制：結果依賴模型與重要性定義。'])], columns=3, footer='特徵選擇必須在訓練資料內進行，避免資料洩漏。'))
add(4, '03_categorical_encoding', cards('類別特徵編碼：有順序與沒有順序', [
    ('有序類別 → 順序編碼', ['尺寸：Small → 0，Medium → 1，Large → 2。', '保留原本存在的先後順序。', '注意：整數間距不一定等於實際程度差。', '模型如何使用這些數值，仍需依任務確認。']),
    ('無序類別 → One-Hot', ['欄位依序為：紅、綠、藍。', '紅 → [1, 0, 0]；綠 → [0, 1, 0]。', '藍 → [0, 0, 1]。', '各欄只表示是否屬於某個類別。', '避免讓模型誤以為顏色有大小關係。'])], footer='類別很多時，要考量稀疏矩陣、記憶體與替代編碼方式。'))

add(5, '02_hyperparameters', cards('超參數：控制模型結構與訓練方式', [
    ('學習率 Learning Rate', ['控制一次參數更新的步幅。', '太小可能收斂慢；太大可能震盪或發散。']),
    ('最大深度 Max Depth', ['控制決策樹可切分到多深。', '影響模型容量；需搭配驗證表現選擇。']),
    ('批次大小 Batch Size', ['控制一次更新使用多少筆樣本。', '常見例子：32、64、128。']),
    ('正則化強度 λ', ['限制模型權重或複雜度。', '太強可能欠擬合；太弱可能過度擬合。'])], footer='使用驗證集或交叉驗證搜尋設定；不要用測試集調參。'))
add(5, '03_parameters_vs_hyperparameters', cards('模型參數與超參數：誰決定？何時決定？', [
    ('模型參數 Parameters', ['由模型在訓練時依資料學出。', '例子：權重 w、偏置 b。', '透過損失與演算法反覆更新。', '訓練後保存於模型檔案。']),
    ('超參數 Hyperparameters', ['由工程師或搜尋程序設定。', '例子：學習率、樹深度、Batch Size。', '控制模型結構、訓練與約束。', '依驗證結果比較不同設定。'])], footer='外層：選超參數 → 內層：訓練模型参数 → 驗證並比較。'.replace('参数','參數')))
add(6, '03_batch_size_comparison', cards('Batch Size：一次更新使用多少資料？', [
    ('較大的批次', ['以更多樣本估計平均梯度，通常較平穩。', '可能提高硬體利用率，但需要更多記憶體。', '同一 Epoch 的更新次數較少。', '不保證比小批次更好的泛化表現。']),
    ('較小的批次', ['梯度估計通常較有波動。', '記憶體需求較低，同一 Epoch 更新較多次。', '可能有助探索，也可能使訓練不穩定。', '需一起調整學習率與訓練預算。'])], footer='SGD：1 筆；Mini-batch：部分資料；Full-batch：全部訓練資料。'))
add(7, '04_naive_bayes', cards('樸素貝氏：以條件機率進行分類', [
    ('核心概念', ['P(類別｜特徵) ∝ P(類別) × P(特徵｜類別)。', '樸素假設：給定類別後，各特徵條件獨立。', '比較各類別的後驗機率，選擇較高者。']),
    ('三種常見模型', ['GaussianNB：連續特徵，以常態分布建模。', 'MultinomialNB：計數特徵，例如詞頻。', 'BernoulliNB：二元特徵，例如詞彙有／無。'])], footer='機率模型的選擇必須符合特徵型態；假設不代表現實一定成立。'))

# Ensemble topology uses explicit branch-and-merge diagrams.
def ensemble(kind):
    s = SVG({'bag':'Bagging：有放回抽樣、獨立訓練、整合預測', 'stack':'Stacking：基礎模型的預測交給元模型'}[kind], 800)
    s.card(35,305,200,175,'輸入資料 X',['訓練特徵與標籤。'],0)
    for i in range(3):
        y=85+i*195
        title=f'抽樣 {i+1} → 模型 {i+1}' if kind=='bag' else ['KNN 基礎模型','SVM 基礎模型','樹狀基礎模型'][i]
        lines=['Bootstrap 有放回抽樣。','各模型獨立訓練。'] if kind=='bag' else ['以交叉驗證產生','OOF 預測特徵。']
        s.card(280,y,340,170,title,lines,i)
        s.arrow(237,390,278,y+85)
        s.arrow(622,y+85,690,390)
    s.card(695,285,275,215,'投票／平均' if kind=='bag' else '元學習器',
           ['分類：多數決。','回歸：數值平均。'] if kind=='bag' else ['輸入：OOF 預測。','目標：對應標籤 y。','學習如何整合預測。'],3)
    s.arrow(973,390,1018,390)
    s.card(1020,325,150,135,'預測 ŷ',['整合結果。'],1)
    s.text(600,710,'Bagging：常用於降低變異數；隨機森林另加入特徵抽樣。' if kind=='bag' else 'OOF：每筆訓練樣本的預測，來自未用該筆樣本訓練的基礎模型。',20,center=True)
    s.text(600,755,'並非保證所有資料上都優於單一模型。' if kind=='bag' else '元模型需要 y 作為訓練目標；避免洩漏的是基礎模型的預測產生方式。',20,center=True)
    return s

add(8,'01_bagging',ensemble('bag'))
add(8,'02_boosting',flow('Boosting：後一輪補強前一輪的不足',[
    ('第 1 輪：基礎模型',['先建立一個弱學習器。','計算目前預測的誤差。']),
    ('第 2 輪：修正不足',['AdaBoost：提高錯誤樣本權重。','梯度提升：擬合負梯度。','平方損失時負梯度對應殘差。']),
    ('後續輪次與整合',['依序建立新的弱學習器。','以學習率控制每輪貢獻。','加總各輪的預測結果。'])],footer='後一輪依賴前一輪的結果；需控制複雜度，避免追逐雜訊。'))
add(8,'03_stacking',ensemble('stack'))

s=SVG('混淆矩陣：真實標籤 y 與預測類別 ŷ',620)
s.text(730,100,'預測類別 ŷ',25,center=True,bold=True)
s.text(515,145,'預測正類（+）',22,center=True)
s.text(925,145,'預測負類（−）',22,center=True)
s.text(125,280,'真實正類（+）',22,center=True)
s.text(125,460,'真實負類（−）',22,center=True)
for x,y,t,lines,col in [(300,170,'TP：真正例',['猜正類，實際也是正類。','例如：患病者被判定患病。'],1),(710,170,'FN：偽負例／漏報',['猜負類，實際是正類。','例如：患病者被判定健康。'],2),(300,350,'FP：偽正例／誤報',['猜正類，實際是負類。','例如：健康者被判定患病。'],2),(710,350,'TN：真負例',['猜負類，實際也是負類。','例如：健康者被判定健康。'],1)]:s.card(x,y,390,160,t,lines,col)
s.text(600,570,'本圖以列表示真實類別、欄表示預測類別；正類的定義由任務決定。',21,center=True)
add(9,'01_confusion_matrix',s)
add(9,'02_classification_metrics',cards('分類指標：四種不同的觀察角度',[
    ('Accuracy：準確率',['(TP + TN) / (TP + TN + FP + FN)','全部樣本中，有多少比例猜對？','類別不平衡時，需搭配其他指標。']),
    ('Precision：精確率',['TP / (TP + FP)','預測為正類的樣本中，有多少是真的？','關注誤報 FP。']),
    ('Recall：召回率',['TP / (TP + FN)','真正的正類中，有多少被找出？','關注漏報 FN。']),
    ('F1：精確率與召回率的調和平均',['2 × Precision × Recall / (Precision + Recall)','也可寫成 2TP / (2TP + FP + FN)。','需依任務成本評估，不能只看單一分數。'])],footer='分母為零時，指標未定義；使用工具的預設處理方式須另行確認。'))

add(10,'05_solutions_overview',cards('模型表現的改善方向：欠擬合與過度擬合',[
    ('欠擬合：訓練與驗證誤差都高',['增加模型容量，捕捉非線性規律。','補充有用特徵與交互特徵。','降低過強的正則化約束。','檢查是否尚未完成收斂。']),
    ('過度擬合：訓練低、驗證高',['加入 L1／L2 正則化或 Dropout。','簡化模型或減少無關特徵。','增加有效資料或合理的資料擴增。','依驗證表現設定 Early Stopping。'])],footer='先確認資料品質、切分方式與評估流程，再比較改善效果。'))
add(11,'01_generalization_concept',flow('泛化：把學到的規律用在未見資料',[
    ('訓練資料：已知 X 與 y',['模型從已知範例學習。','不能只記住資料中的雜訊。']),
    ('模型：學習可用的規律',['容量太小可能欠擬合。','容量過高可能過度擬合。','用驗證資料協助選擇。']),
    ('未見資料：新的 X',['產生預測 ŷ。','用對應 y 評估表現。','良好泛化需要誤差低。'])],footer='訓練與驗證的差距小不代表模型好；還要看兩者是否都低。'))
add(11,'03_data_splitting',cards('資料分割：先保留最後的客觀驗收',[
    ('Train：60%～70%（示例）',['學習權重與其他模型參數。','預處理、特徵選擇只在 Train fit。']),
    ('Validation：15%～20%（示例）',['調整超參數、挑模型與早停。','不能替代最終測試集。']),
    ('Test：15%～20%（示例）',['模型定型後再進行最終評估。','不得用於訓練或挑選設定。'])],columns=3,footer='比例須合計 100%；訓練得到的預處理規則才可應用到其他子集。'))

s=SVG('K 折交叉驗證：每一折輪流驗證，其餘折訓練',650)
for i in range(5):
    s.text(65,150+i*70,f'第 {i+1} 輪',20)
    for j in range(5):
        s.rect(175+j*175,115+i*70,155,48,2 if i==j else 0)
        s.text(252+j*175,146+i*70,'驗證' if i==j else '訓練',22,center=True)
s.text(600,525,'5-Fold：得到 5 個驗證分數，觀察其平均與變動程度。',23,center=True)
s.text(600,565,'每一輪都重新擬合模型與預處理；此圖是一般 K 折的概念示意。',21,center=True)
s.text(600,605,'先保留獨立 Test；時間序列需使用保持時間順序的驗證方式。',21,center=True)
add(11,'04_cross_validation',s)
add(11,'05_improve_generalization',cards('提升泛化：四個改善方向',[
    ('資料與特徵',['增加有代表性的有效資料。','合理資料擴增，維持標籤語意。','檢查標籤品質與特徵資訊。']),
    ('模型容量',['選擇合適的深度與複雜度。','必要時剪枝、簡化模型。','比較集成方法的驗證表現。']),
    ('正則化約束',['L1／L2：限制模型權重。','Dropout：訓練時隨機停用部分單元。','約束太強也可能欠擬合。']),
    ('訓練與評估',['用驗證資料調設定與早停。','選擇合適的優化器和學習率。','以交叉驗證檢查穩定性。'])],footer='改善需要驗證；不要用最終測試集反覆挑選方案。'))
add(12,'01_ml_lifecycle_pipeline',flow('機器學習專案的六個階段',[
    ('1. 問題定義',['確認學習類型與商業目標。','定義 KPI 與 Baseline。']),
    ('2. 資料收集與探索',['清洗、去重、檢查缺失。','探索分布與標籤品質。']),
    ('3. 切分與特徵工程',['先分 Train／Val／Test。','縮放、編碼與特徵選擇。','所有 fit 僅用 Train。']),
    ('4. 模型選型與訓練',['選擇模型或集成方法。','執行訓練與參數更新。']),
    ('5. 評估與調整',['Validation：診斷與調參。','Test：最後的泛化驗收。']),
    ('6. 部署與監控',['API、服務或裝置推論。','監控效能與資料漂移。','必要時回到資料與訓練。'])],footer='依編號順序進行；部署後的監控會回饋到新一輪資料與訓練。'))

s=SVG('模型訓練迴圈：一個批次完成四個步驟',800)
positions=[(40,95),(640,95),(640,370),(40,370)]
items=[('1. 前向傳播 Forward Pass',['輸入批次特徵 X 與目前參數。','計算模型預測 ŷ。']),('2. 計算損失 Compute Loss',['比較預測 ŷ 與真實標籤 y。','例如：MSE 或 Cross-Entropy。']),('3. 反向傳播 Backpropagation',['以連鎖律計算損失對參數的梯度。','求得 ∇J(θ)。']),('4. 參數更新 Parameter Update',['依優化器規則更新權重。','例如 SGD：θ ← θ − α∇J(θ)。'])]
for i,((x,y),(t,lines)) in enumerate(zip(positions,items)):s.card(x,y,520,195,t,lines,i)
s.arrow(565,195,635,195);s.arrow(900,295,900,365);s.arrow(635,465,565,465);s.arrow(300,365,300,295)
s.text(600,635,'Iteration：完成一批資料的參數更新；Batch Size：這一批的樣本數。',22,center=True)
s.text(600,685,'Epoch：走過一次訓練資料；保留最後不足一批時，更新次數 = ceil(N / Batch Size)。',22,center=True)
s.text(600,735,'例：N = 10,000、Batch Size = 64 → 每 Epoch 157 次；10 Epochs 共 1,570 次。',22,center=True)
add(12,'02_model_training_iteration_loop',s)

(BASE/'tools/svg_map.json').write_text(json.dumps(specs,ensure_ascii=False,indent=2)+'\n')
print(f'Redrawn {len(specs)} SVG diagrams; original PNG files preserved.')
