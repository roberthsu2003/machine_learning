"""Build concise student starts for project READMEs and the project catalog.

The detailed teaching notes in each README are preserved inside a collapsible
section. Edit PROJECTS below to change the student-facing route or outcome.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
from urllib.parse import quote
import re

ROOT = Path(__file__).resolve().parent
GUIDED = 'AI提問與機器學習實戰'
WORK = 'AI協作機器學習_職場實戰篇'
GITHUB = 'https://github.com/roberthsu2003/machine_learning/blob/main/'
DETAIL_BEGIN = '<!-- student-guide:details -->'
DETAIL_END = '<!-- student-guide:details-end -->'

# series, directory, task, level, question, outcome
PROJECTS = [
 (GUIDED,'01_廣告效益預測_迴歸篇','迴歸','入門','廣告投入能預測多少銷售額？','辨認數值目標，解讀 MAE 與 R²。'),
 (GUIDED,'02_鐵達尼號生存預測_分類篇','分類','入門','乘客是否生還？','處理缺值並讀懂混淆矩陣。'),
 (GUIDED,'03_商場客戶分群_集群篇','分群','入門','顧客可以分成哪些群？','理解沒有標籤的分群與 K 值。'),
 (GUIDED,'04_葡萄酒品種辨識_經典分類演算法篇','分類','進階','哪種模型較適合辨識葡萄酒？','用交叉驗證比較多種分類模型。'),
 (GUIDED,'05_乳癌診斷_PCA降維篇','分類／降維','進階','降維後的分類結果如何變化？','比較 PCA 前後的資料與模型；僅供教學。'),
 (GUIDED,'06_商場客群輪廓探索_進階分群篇','分群','進階','不同分群方法會給出什麼結果？','比較階層式分群、DBSCAN 與雜訊點。'),
 (GUIDED,'07_葡萄酒品種辨識_神經網路篇','分類','進階','神經網路一定比較好嗎？','與基準模型比較並觀察訓練曲線。'),
 (WORK,'01_業務_潛在客戶成交預測','分類','入門','哪筆商機值得先跟進？','分清特徵與標籤，解讀成交機率。'),
 (WORK,'02_電商_商品退貨風險預測','分類','入門','哪筆訂單可能退貨？','處理缺值與類別欄位。'),
 (WORK,'03_行銷_廣告預算效益預測','迴歸','入門','廣告投入可能帶來多少營收？','用誤差指標檢查數值預測。'),
 (WORK,'04_選品_二手商品定價估算','迴歸','入門','二手商品可以如何估價？','看懂多特徵輸入與殘差。'),
 (WORK,'05_人資_員工離職風險預警','分類','進階','模型如何辨認離職風險？','比較樹模型與交叉驗證；僅供教學。'),
 (WORK,'06_零售_顧客消費價值分群','分群','入門','顧客消費型態有何差異？','用 RFM 特徵理解顧客分群。'),
 (WORK,'07_餐飲_門市每日營收預估','迴歸','進階','明天門市營收可能是多少？','練習用多個營運特徵預測。'),
 (WORK,'08_訂閱_會員流失提早挽回','分類','進階','哪些會員可能不續約？','理解類別不平衡與漏判。'),
 (WORK,'09_物流_宅配包裹延遲預警','分類','進階','哪些包裹可能延遲？','學習群組分割與避免資料洩漏。'),
 (WORK,'10_門市_烘焙生鮮備料預估','迴歸','進階','今天該準備多少份商品？','只用預測當下已知的時間資料。'),
 (WORK,'11_客服_顧客評論情緒分析','文字分類','進階','哪些評論需要先處理？','將文字轉成特徵並檢查分類結果。'),
 (WORK,'12_營運_促銷活動成效監控','分類／監控','進階','模型上線後何時可能失準？','辨認資料漂移與監控需求。'),
 (WORK,'13_法務部_受刑人假釋再犯預測','分類','進階','示範資料如何建立風險分類？','檢查漏判、誤報與模型限制；不可用於實際審查。'),
 (WORK,'14_法務部_公法欠費徵起預測','分類','進階','示範資料如何預測徵起結果？','理解案件特徵與分類評估；不可用於實際執行。'),
 (WORK,'15_法務部_涉詐人頭帳戶預警','分類','進階','示範資料如何標記可疑帳戶？','檢查錯誤分類與資料限制；不可用於實際處置。'),
]


def title(directory: str) -> str:
    return directory.split('_',1)[1].replace('_','｜')


def project_files(folder: Path):
    notebooks=sorted(folder.glob('*.ipynb'))
    csvs=sorted(folder.glob('*.csv'))
    assert len(notebooks)==1, f'Expected one notebook in {folder}'
    assert csvs, f'Missing CSV in {folder}'
    return notebooks[0],csvs[0],folder/'app.py'


def write_project_readme(series, directory, task, level, question, outcome):
    folder=ROOT/series/directory
    readme=folder/'README.md'
    original=readme.read_text(encoding='utf-8')
    if DETAIL_BEGIN in original:
        detail=original.split(DETAIL_BEGIN,1)[1].split(DETAIL_END,1)[0].strip()
    else:
        detail=re.sub(r'^# .+\n','',original,count=1).strip()
    notebook,csv,app=project_files(folder)
    number=directory.split('_',1)[0]
    name=title(directory)
    notice='\n> 本專案使用教學示範資料，不能用於個人風險評估或實際公務決策。\n' if number in {'13','14','15'} and series==WORK else ''
    if series==GUIDED and number=='05':
        notice='\n> 本專案僅供機器學習教學，不能用於醫療診斷。\n'
    student=f'''# {number}｜{name}

**任務：{task} · 程度：{level}**

**要回答的問題：{question}**

## 從這裡開始

1. 開啟 [教學 Notebook](./{quote(notebook.name)})，從第一個儲存格依序執行。
2. 對照 [資料集](./{quote(csv.name)})，找出輸入欄位與預測目標。
3. 完成後再看 [示範應用程式](./app.py)，比較它的輸入、輸出與 Notebook 是否一致。

**完成檢查：**{outcome}
{notice}
[返回本系列目錄](../README.md) · [回到全部實戰專案](../../docs/projects/index.html)

<details>
<summary>展開完整說明、資料欄位與 AI 提問</summary>

{DETAIL_BEGIN}

{detail}

{DETAIL_END}
</details>
'''
    readme.write_text(student,encoding='utf-8')


def project_card(project):
    series,directory,task,level,question,outcome=project
    folder=ROOT/series/directory
    notebook,_,_=project_files(folder)
    gh_dir=GITHUB+quote(f'{series}/{directory}/README.md')
    gh_nb=GITHUB+quote(f'{series}/{directory}/{notebook.name}')
    return f'''<article class="card"><div class="card-meta"><span>{escape(directory[:2])}</span><span>{escape(task)}</span><span>{escape(level)}</span></div><h3>{escape(title(directory))}</h3><p>{escape(question)}</p><p class="outcome">學習重點：{escape(outcome)}</p><div class="card-actions"><a href="{gh_dir}">看專案說明 →</a><a href="{gh_nb}">開 Notebook ↗</a></div></article>'''


def write_site():
    guided=''.join(project_card(p) for p in PROJECTS if p[0]==GUIDED)
    work=''.join(project_card(p) for p in PROJECTS if p[0]==WORK)
    html=f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><link rel="icon" href="../favicon.svg" type="image/svg+xml"><title>選擇實戰專案｜機器學習學生路線</title><meta name="description" content="依學習目的選擇 22 個機器學習範例專案。"><style>
:root{{--ink:#16322e;--green:#176d59;--muted:#58716a;--line:#d7e5dd;--paper:#f7faf7;--white:#fff}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:system-ui,-apple-system,"Noto Sans TC",sans-serif;line-height:1.65}}a{{color:var(--green)}}header,main,footer{{max-width:1180px;margin:auto;padding:24px}}header{{display:flex;justify-content:space-between;gap:20px;border-bottom:1px solid var(--line)}}header a{{font-weight:700;text-decoration:none}}nav{{display:flex;gap:18px;flex-wrap:wrap}}.hero{{padding:68px 0 55px}}.eyebrow,.number{{font-weight:750;color:var(--green);letter-spacing:.06em}}h1{{font-size:clamp(2.4rem,5vw,4.1rem);line-height:1.18;margin:12px 0}}.hero p{{font-size:1.15rem;color:var(--muted);max-width:710px}}.route{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:32px 0}}.route div{{padding:20px;background:#e8f2ec;border-radius:12px}}.route strong{{display:block;font-size:1.1rem}}section.track{{margin:58px 0}}.section-head{{border-top:3px solid var(--green);padding-top:18px;margin-bottom:20px}}.section-head h2{{font-size:2rem;margin:5px 0}}.section-head p{{color:var(--muted);margin:4px 0}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}}.card{{background:var(--white);border:1px solid var(--line);border-radius:14px;padding:22px;display:flex;flex-direction:column;min-height:265px}}.card-meta{{display:flex;gap:7px;flex-wrap:wrap}}.card-meta span{{font-size:.79rem;font-weight:700;background:#edf4ee;color:var(--green);padding:3px 8px;border-radius:99px}}.card h3{{font-size:1.22rem;line-height:1.35;margin:14px 0 7px}}.card p{{margin:5px 0}}.outcome{{color:var(--muted);font-size:.94rem}}.card-actions{{margin-top:auto;padding-top:18px;display:flex;gap:15px;flex-wrap:wrap;font-weight:700}}.notice{{background:#fff5e7;padding:18px;border-radius:10px}}footer{{border-top:1px solid var(--line);color:var(--muted)}}@media(max-width:900px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:620px){{header{{display:block}}nav{{margin-top:10px}}.route,.grid{{grid-template-columns:1fr}}.hero{{padding:48px 0}}}}
</style></head><body><header><a href="../">徐國堂 Python 實作應用班</a><nav aria-label="導覽"><a href="../">15 頁入門講義</a><a href="#guided">AI 提問實戰</a><a href="#workplace">職場實戰</a><a href="../glossary/">名詞動畫</a></nav></header><main><div class="hero"><div class="eyebrow">實戰專案目錄 · 22 個範例</div><h1>先選一個問題，<br>再動手跑資料。</h1><p>初學者先完成一個迴歸、一個分類與一個分群專案。每個專案都從 Notebook 開始，資料與示範程式放在同一資料夾。</p><div class="route"><div><strong>① 迴歸</strong>廣告效益預測：看懂數值預測與誤差。</div><div><strong>② 分類</strong>鐵達尼號：看懂分類結果與混淆矩陣。</div><div><strong>③ 分群</strong>商場客戶：探索沒有答案標籤的資料。</div></div></div><section class="track" id="guided"><div class="section-head"><span class="number">路線 A · 7 個專案</span><h2>AI 提問與機器學習實戰</h2><p>從經典資料集入門，學會閱讀 Notebook、比較模型與解釋結果。建議先做 01 → 02 → 03。</p></div><div class="grid">{guided}</div></section><section class="track" id="workplace"><div class="section-head"><span class="number">路線 B · 15 個專案</span><h2>AI 協作機器學習｜職場實戰</h2><p>依工作情境選題；初次接觸可先做 01、03 或 06，再挑自己的領域。</p></div><div class="grid">{work}</div></section><p class="notice">司法與醫療主題使用教學範例，僅用來練習模型流程與錯誤分析，不供實際個人判斷或處置。</p></main><footer>機器學習學生路線 · <a href="../">返回入門講義</a></footer></body></html>'''
    html=html.replace('.route div{padding:20px;background:#e8f2ec;border-radius:12px}',
                      '.route a{display:block;padding:20px;background:#e8f2ec;border-radius:12px;text-decoration:none;color:inherit}.route a:hover,.route a:focus-visible{background:#d8e9de;outline:2px solid var(--green)}')
    route_projects=[p for p in PROJECTS if p[0]==GUIDED][:3]
    for number,label,description,project in zip(
        ('① 迴歸','② 分類','③ 分群'),
        ('廣告效益預測','鐵達尼號','商場客戶'),
        ('看懂數值預測與誤差。','看懂分類結果與混淆矩陣。','探索沒有答案標籤的資料。'),
        route_projects,
    ):
        old=f'<div><strong>{number}</strong>{label}：{description}</div>'
        target=GITHUB+quote(f'{project[0]}/{project[1]}/README.md')
        new=f'<a href="{target}"><strong>{number}</strong>{label}：{description}</a>'
        assert old in html
        html=html.replace(old,new,1)
    dest=ROOT/'docs'/'projects'
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'index.html').write_text(html,encoding='utf-8')


def write_series_readmes():
    for series in (GUIDED,WORK):
        chosen=[p for p in PROJECTS if p[0]==series]
        guided=series==GUIDED
        heading='AI 提問與機器學習實戰' if guided else 'AI 協作機器學習｜職場實戰篇'
        summary=('7 個經典資料集專案，適合學完入門講義後練習閱讀 Notebook、比較模型與解釋結果。'
                 if guided else '15 個工作情境專案，依你的領域選題。每個資料夾都有 Notebook、CSV 與示範 app.py。')
        route=('01 廣告效益預測 → 02 鐵達尼號生存預測 → 03 商場客戶分群'
               if guided else '01 業務成交預測 → 03 廣告預算效益預測 → 06 顧客價值分群')
        rows=[]
        for p in chosen:
            _,directory,task,level,question,_=p
            num=directory[:2]
            rows.append(f'| {num} | [{title(directory)}](./{quote(directory)}/README.md) | {task} | {level} | {question} |')
        warning=('\n> 司法案件與人員風險主題只供教學，不可用於實際個人判斷或處置。\n'
                 if not guided else '\n> 乳癌資料範例只供模型教學，不可用於醫療診斷。\n')
        teacher=('\n教師若要指定 AI 協作規範，可參考 [AGENT_TEMPLATE.md](./AGENT_TEMPLATE.md)；學生先從已附的 Notebook 開始即可。\n'
                 if not guided else '')
        readme=f'''# {heading}

{summary}

## 第一次來，先做這三個

**{route}**

每做完一個專案，試著回答：要預測什麼？輸入資料有哪些？用什麼指標檢查？模型在哪些情況下可能失準？

## 怎麼使用一個專案

1. 點進下表選一個情境，先看專案頁頂端的「從這裡開始」。
2. 開啟 Notebook，依序執行並對照同資料夾的 CSV。
3. 看懂結果後再開 `app.py`；比較介面輸入、模型輸出與 Notebook 的結果。

## 專案目錄

| 編號 | 專案 | 任務 | 程度 | 要回答的問題 |
| :--- | :--- | :--- | :--- | :--- |
{chr(10).join(rows)}
{warning}
進一步查詢名詞時使用 [互動名詞教材](../docs/glossary/index.html)。全部專案也可在 [視覺化目錄](../docs/projects/index.html) 按情境瀏覽。
{teacher}
[返回教材總入口](../README.md)
'''
        (ROOT/series/'README.md').write_text(readme,encoding='utf-8')


def main():
    for p in PROJECTS:write_project_readme(*p)
    write_site()
    write_series_readmes()


if __name__=='__main__':main()
