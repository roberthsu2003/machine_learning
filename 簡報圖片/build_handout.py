"""Build the 15-page student introduction in PDF, PPTX, PNG, SVG, and HTML.

The supplied PowerPoint is image based. SVGs in source/ are editable slide
sources; source/base.pptx is the untouched slide package used for export.
Requires rsvg-convert, sips, and pdfunite.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SOURCE = ROOT / 'source'
DOCS = REPO / 'docs'
W, H = 2000, 1125
BG = '#081128'
PANEL = '#142443'
WHITE = '#f5f8ff'
MUTED = '#b4c0d4'
CYAN = '#3bc9f0'
GREEN = '#30ddb0'
PURPLE = '#ac83ff'
ORANGE = '#ffbd64'
PINK = '#ff6eab'
MODULES = [('01 概念地圖', CYAN), ('02 模型流程', GREEN), ('03 動手預測', ORANGE)]


def text(x, y, value, size=42, color=WHITE, weight=400, anchor='start'):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>')


def rect(x, y, w, h, fill=PANEL, stroke=None, radius=18, width=3):
    stroke = stroke or fill
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def line(x1, y1, x2, y2, color=MUTED, width=3):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>'


def circle(cx, cy, r, fill, stroke=None, width=3):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke or fill}" stroke-width="{width}"/>'


def page(n, title, subtitle='', module=0):
    accent = MODULES[module][1]
    a = [rect(0, 0, W, H, BG, BG, 0, 0),
         text(130, 75, MODULES[module][0], 28, accent, 700),
         text(1860, 75, f'{n:02d} / 15', 27, MUTED, 400, 'end'),
         text(130, 175, title, 58, WHITE, 700),
         line(130, 218, 1870, 218, accent, 3)]
    if subtitle:
        a.append(text(130, 270, subtitle, 31, MUTED))
    return a


def foot(a, value, accent=CYAN):
    a += [line(130, 985, 1870, 985, '#354660', 2), text(130, 1043, value, 33, accent, 600)]
    return a


def svg(parts):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="2000" height="1125" '
            'viewBox="0 0 2000 1125"><g font-family="Heiti TC, PingFang TC, sans-serif">'
            + ''.join(parts) + '</g></svg>')


def slide1():
    a = [rect(0,0,W,H,BG,BG,0,0), text(130,100,'徐國堂 Python 實作應用班',34,CYAN,600),
         text(130,335,'機器學習，從資料到預測',78,WHITE,700),
         text(130,430,'一份給初學者的 15 頁學習路線',39,MUTED),
         line(130,500,1870,500,CYAN,4)]
    for i,(num,label,detail,color) in enumerate([
        ('01','看懂概念','AI、ML、DL 與問題類型',CYAN),
        ('02','學會流程','資料、訓練、測試與評估',GREEN),
        ('03','做出預測','房價範例與五分鐘練習',ORANGE)]):
        x=130+i*590
        a += [text(x,610,num,49,color,700),text(x,690,label,43,WHITE,700),text(x,755,detail,30,MUTED)]
    a.append(text(130,1020,'學完後，你能說清楚模型如何學習，也能判斷預測值能不能相信。',34,WHITE))
    return a


def slide2():
    a=page(2,'這堂課只走一條路','用房價預測串起所有名詞與步驟。',0)
    items=[('1','提出問題','要預測房價'),('2','整理資料','坪數與成交價'),('3','訓練模型','從例子找規律'),('4','檢查表現','用未見資料測試'),('5','預測新屋','輸出估計價格')]
    for i,(num,name,detail) in enumerate(items):
        x=130+i*350
        a += [circle(x+100,450,53,CYAN),text(x+100,466,num,45,BG,700,'middle'),
              text(x+100,600,name,38,WHITE,700,'middle'),text(x+100,665,detail,27,MUTED,400,'middle')]
        if i<4: a.append(text(x+275,470,'→',57,CYAN,600,'middle'))
    return foot(a,'順序記法：問題 → 資料 → 訓練 → 檢查 → 預測')


def slide3():
    a=page(3,'AI、ML、DL：看範圍；生成式 AI：看任務','生成式 AI 多使用深度學習，但不是固定的第四層。',0)
    a += [rect(130,330,1050,575,'#162441',CYAN,250,5),
          rect(250,445,810,360,'#173a40',GREEN,175,5),
          rect(380,555,550,160,'#402855',PURPLE,80,5),
          text(455,405,'人工智慧 AI',43,CYAN,700),text(470,520,'機器學習 ML',42,GREEN,700),
          text(650,655,'深度學習 DL',39,PURPLE,700,'middle'),
          rect(1260,385,610,470,'#2a1e3c',PINK,22,4),
          text(1320,475,'生成式 AI',49,PINK,700),
          text(1320,560,'任務：產生新內容',34,WHITE),
          text(1320,630,'例：文字、圖片',34,WHITE),
          text(1320,720,'常見方法：深度學習',31,MUTED)]
    return foot(a,'例：決策樹屬於 ML；神經網路屬於 DL；文字生成是 GenAI 任務。')


def slide4():
    a=page(4,'機器學習做了什麼？','以監督式學習中的「判斷垃圾郵件」為例。',0)
    a += [rect(130,350,800,500,stroke=CYAN),rect(1070,350,800,500,stroke=GREEN),
          text(190,450,'寫規則',48,CYAN,700),text(190,545,'人先決定判斷條件',37,WHITE),
          text(190,635,'例如：標題含「中獎」就攔截',32,MUTED),
          text(1130,450,'用資料學習',48,GREEN,700),text(1130,545,'給模型郵件與正確標籤',37,WHITE),
          text(1130,635,'由模型調整判斷方式',32,MUTED)]
    return foot(a,'機器學習需要清楚的問題、可用資料，以及檢查結果的方法。')


def slide5():
    a=page(5,'先問：你要預測哪一種答案？','問題類型決定資料、模型與評估方式。',1)
    a += [rect(130,345,800,520,stroke=CYAN),rect(1070,345,800,520,stroke=ORANGE),
          text(190,450,'分類',51,CYAN,700),text(190,550,'答案是類別',39,WHITE),
          text(190,640,'例：貓或狗、垃圾郵件或正常郵件',32,MUTED),
          text(190,755,'輸出：類別或各類別的機率',31,CYAN),
          text(1130,450,'迴歸',51,ORANGE,700),text(1130,550,'答案是數值',39,WHITE),
          text(1130,640,'例：房價、每日營收',32,MUTED),
          text(1130,755,'輸出：一個估計數值',31,ORANGE)]
    return foot(a,'本講義用房價迴歸示範完整流程。',GREEN)


def slide6():
    a=page(6,'每一筆資料都有輸入 X 與答案 y','房價例子：坪數是特徵 X；成交價是標籤 y。',1)
    a += [rect(130,345,1740,510,stroke=GREEN),
          text(250,430,'房屋',35,GREEN,700),text(700,430,'坪數 X',35,GREEN,700),text(1300,430,'成交價 y',35,GREEN,700)]
    rows=[('#101','18 坪','180 萬元'),('#102','50 坪','500 萬元'),('#103','95 坪','950 萬元')]
    for i,row in enumerate(rows):
        y=520+i*105
        a += [line(220,y-45,1780,y-45,'#354660',2),text(250,y,row[0],36,WHITE),
              text(700,y,row[1],36,WHITE),text(1300,y,row[2],36,WHITE)]
    return foot(a,'訓練時給模型 X 和 y；真正預測時只有新房屋的 X。',GREEN)


def slide7():
    a=page(7,'先分資料，才開始訓練','把測試資料留到最後，檢查模型遇到新資料的表現。',1)
    a += [rect(130,360,1740,165,stroke=GREEN),text(1000,465,'全部已知資料',45,WHITE,700,'middle'),
          text(1000,600,'↓',55,GREEN,700,'middle'),
          rect(130,660,780,215,stroke=CYAN),rect(1090,660,780,215,stroke=ORANGE),
          text(520,748,'訓練資料',43,CYAN,700,'middle'),text(520,815,'用來學習規律',31,MUTED,400,'middle'),
          text(1480,748,'測試資料',43,ORANGE,700,'middle'),text(1480,815,'留到最後檢查',31,MUTED,400,'middle')]
    return foot(a,'測試資料的答案不能提前參與訓練，否則分數可能失真。',GREEN)


def slide8():
    a=page(8,'訓練，就是反覆減少預測誤差','模型不是「記住答案」；它要學到能用於新資料的規律。',1)
    steps=[('1','輸入訓練資料'),('2','模型給出預測'),('3','比較預測與答案'),('4','調整模型參數')]
    for i,(num,label) in enumerate(steps):
        x=130+i*445
        a += [rect(x,390,390,225,stroke=GREEN),text(x+195,470,num,42,GREEN,700,'middle'),
              text(x+195,555,label,34,WHITE,600,'middle')]
        if i<3:a.append(text(x+418,515,'→',45,GREEN,700,'middle'))
    a += [text(1000,740,'重複多次，再用測試資料檢查。',43,CYAN,700,'middle')]
    return foot(a,'訓練誤差低，不等於新資料一定預測得準。',GREEN)


def slide9():
    a=page(9,'分類例子：模型怎麼判斷貓或狗？','這是一筆資料的教學示意，不代表完整影像辨識系統。',1)
    a += [rect(130,350,500,390,stroke=CYAN),rect(750,350,500,390,stroke=PURPLE),rect(1370,350,500,390,stroke=GREEN),
          text(380,450,'輸入 X',44,CYAN,700,'middle'),text(380,550,'一張狗的照片',37,WHITE,400,'middle'),
          text(1000,450,'模型預測',44,PURPLE,700,'middle'),text(1000,545,'貓 80%',39,WHITE,400,'middle'),text(1000,620,'狗 20%',39,WHITE,400,'middle'),
          text(1620,450,'正確答案 y',44,GREEN,700,'middle'),text(1620,550,'狗',48,WHITE,700,'middle'),
          text(1000,835,'預測與答案不同 → 計算誤差 → 調整參數',34,ORANGE,600,'middle')]
    return foot(a,'分類模型常輸出各類別的機率，再選擇預測類別。',GREEN)


def slide10():
    a=page(10,'迴歸例子：用坪數估計房價','以下六筆資料與直線只用於教學，單位為萬元。',1)
    rows=[(18,180),(23,200),(42,400),(50,500),(66,650),(95,950)]
    a += [rect(130,330,675,575,stroke=GREEN),text(190,405,'坪數 X',34,GREEN,700),text(500,405,'成交價 y',34,GREEN,700),
          rect(880,330,990,575,stroke=CYAN),text(930,415,'示意直線：y = 10.2 × X − 15',38,CYAN,700),
          line(970,835,1760,835,MUTED,2),line(970,490,970,835,MUTED,2),line(985,826,1760,490,PINK,5),
          text(1635,883,'坪數 X',27,MUTED),text(980,475,'價格 y',27,MUTED)]
    for i,(x,y) in enumerate(rows):
        yy=475+i*65
        a += [text(200,yy,str(x),33,WHITE),text(520,yy,str(y),33,WHITE)]
        px=970+x/100*790; py=835-y/1000*345
        a.append(circle(f'{px:.1f}',f'{py:.1f}',10,CYAN,WHITE,2))
    return foot(a,'模型找整體規律，直線不必穿過每一個資料點。',GREEN)


def slide11():
    a=page(11,'如何知道預測好不好？','先看測試資料，再選與問題相符的指標。',1)
    a += [rect(130,350,800,495,stroke=CYAN),rect(1070,350,800,495,stroke=ORANGE),
          text(190,445,'分類：答對了多少？',43,CYAN,700),
          text(190,545,'正確率 = 答對筆數 ÷ 總筆數',33,WHITE),
          text(190,650,'類別不平衡時，還要看漏判。',31,MUTED),
          text(1130,445,'迴歸：差了多少？',43,ORANGE,700),
          text(1130,545,'絕對誤差 = |預測 − 真值|',33,WHITE),
          text(1130,650,'平均絕對誤差 MAE：越小越好。',31,MUTED)]
    return foot(a,'訓練分數與測試分數都要看；差距太大可能是過度擬合。',GREEN)


def slide12():
    a=page(12,'房價也會受其他特徵影響','單看坪數容易漏掉重要資訊。',1)
    a += [rect(130,340,1740,510,stroke=GREEN),
          text(225,430,'坪數',32,GREEN,700),text(560,430,'地點',32,GREEN,700),
          text(1010,430,'房數',32,GREEN,700),text(1320,430,'屋齡',32,GREEN,700),text(1590,430,'價格 y',32,ORANGE,700)]
    rows=[('50','A 區','3','5','500'),('66','B 區','4','12','650'),('42','C 區','2','8','400')]
    xs=[225,560,1010,1320,1590]
    for i,row in enumerate(rows):
        y=535+i*100
        a += [line(205,y-45,1790,y-45,'#354660',2)]
        a += [text(xs[j],y,row[j],34,WHITE) for j in range(5)]
    return foot(a,'地點是類別資料，通常要先編碼；多加特徵也要用測試資料驗證。',GREEN)


def slide13():
    a=page(13,'預測新房屋：只有輸入，沒有真實答案','先使用第 10 頁的單特徵示意模型。',1)
    a += [rect(130,375,500,420,stroke=CYAN),rect(750,375,500,420,stroke=PURPLE),rect(1370,375,500,420,stroke=GREEN),
          text(380,465,'新房屋',45,CYAN,700,'middle'),text(380,565,'60 坪',45,WHITE,600,'middle'),
          text(1000,465,'已訓練模型',43,PURPLE,700,'middle'),text(1000,565,'10.2 × 60 − 15',38,WHITE,500,'middle'),
          text(1620,465,'預測價格',43,GREEN,700,'middle'),text(1620,565,'597 萬元',44,WHITE,700,'middle'),
          text(690,590,'→',54,CYAN,700,'middle'),text(1310,590,'→',54,GREEN,700,'middle')]
    return foot(a,'597 是估計值，不是真實成交價；不同模型可能給不同答案。',GREEN)


def slide14():
    a=page(14,'五分鐘練習：你會怎麼預測？','先自己估計，再用 Python 計算。不要先看下一頁。',2)
    a += [rect(130,350,795,520,stroke=ORANGE),rect(1075,350,795,520,stroke=CYAN),
          text(190,440,'已知成交資料',42,ORANGE,700),
          text(190,535,'18坪→180萬　23坪→200萬',31,WHITE),
          text(190,600,'42坪→400萬　50坪→500萬',31,WHITE),
          text(190,665,'66坪→650萬　95坪→950萬',31,WHITE),
          text(190,790,'你估計 72 坪：______ 萬元',35,ORANGE,600),
          text(1135,440,'在 Colab 執行',42,CYAN,700),
          text(1135,540,'x = 72',39,WHITE),
          text(1135,615,'y = 10.2 * x - 15',39,WHITE),
          text(1135,690,'print(y)',39,WHITE),
          text(1135,750,'colab.research.google.com',29,MUTED),
          text(1135,825,'記下輸出：______ 萬元',35,CYAN,600)]
    return foot(a,'比較自己的估計與模型輸出，再說出可能的誤差來源。',ORANGE)


def slide15():
    a=page(15,'答案與離開教室前的三個確認','72 坪代入示意模型：10.2 × 72 − 15 = 719.4 萬元。',2)
    questions=[('1','X 與 y 是什麼？','坪數是 X；成交價是 y。'),
               ('2','何時使用測試資料？','訓練完成後，檢查新資料表現。'),
               ('3','719.4 是真實價格嗎？','不是；只是單特徵模型的估計。')]
    for i,(num,q,ans) in enumerate(questions):
        y=350+i*185
        a += [rect(130,y,1740,145,stroke=ORANGE),text(195,y+92,num,50,ORANGE,700),
              text(315,y+65,q,37,WHITE,700),text(315,y+115,ans,29,MUTED)]
    return foot(a,'下一步：用課程範例練習資料清理、建模與評估。',ORANGE)

SLIDES=[slide1,slide2,slide3,slide4,slide5,slide6,slide7,slide8,slide9,slide10,slide11,slide12,slide13,slide14,slide15]
CONTENT=[
('這堂課要學會什麼','看懂模型從資料學到規律的流程，並判斷預測值的意義。','能區分概念、完成簡單預測、說出限制。'),
('一條學習路線','先提出問題，再整理資料、訓練、檢查，最後預測。','請先記住順序，後面的名詞都能放回這條路線。'),
('AI、ML、DL 與生成式 AI','AI 包含 ML，ML 包含 DL。生成式 AI 描述產生文字或圖片等內容的任務。','生成式 AI 不是固定的第四層；目前常見系統多使用深度學習。'),
('機器學習做了什麼','傳統程式由人寫判斷規則；機器學習用帶答案的例子調整模型。','本課聚焦監督式學習。'),
('先選問題類型','分類預測類別；迴歸預測數值。本課用房價迴歸做主例。','房價是數值，因此屬於迴歸。'),
('認識特徵與標籤','坪數是輸入特徵 X，成交價是答案或標籤 y。預測新房屋時只有 X。','請指出一筆資料中哪些欄位可在預測當下取得。'),
('先分訓練與測試資料','訓練資料用來學習；測試資料留到最後檢查新資料表現。','測試資料的答案不能提前參與訓練。'),
('模型如何訓練','輸入資料、得到預測、比較答案、調整參數；反覆執行。','訓練誤差變小，仍須用測試資料檢查。'),
('分類小例子','狗的照片被預測成貓 80%、狗 20%。比較正確標籤後，模型調整參數。','這是單筆資料的機制示意。'),
('迴歸小例子','用六筆房價資料畫散佈圖，再用示意直線 y = 10.2 × X − 15 估計價格。','直線不必通過每個資料點。'),
('怎麼判斷好壞','分類可先看正確率；迴歸可看平均絕對誤差。兩者都應在測試資料上檢查。','訓練分數好、測試分數差，可能過度擬合。'),
('加入更多特徵','房價還可能受地點、房數和屋齡影響；地點通常需要編碼。','新增特徵是否有幫助，仍要用測試資料驗證。'),
('預測新資料','先用 60 坪示範：把新房屋坪數代入單特徵模型，得到 597 萬元。下一頁換你預測 72 坪。','預測值是估計，不是真實成交價。'),
('五分鐘練習','先估計 72 坪房價，再在 Colab 執行 x = 72、y = 10.2 * x - 15、print(y)。','想想自己的估計與模型輸出為何不同。'),
('答案與檢查','模型輸出 719.4 萬元。確認特徵、標籤、測試資料與預測限制。','這個公式只看坪數，不能當作實際估價。')]


def run(*args):
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


def jpeg_pdf(jpeg: Path, dest: Path, width: int, height: int):
    data=jpeg.read_bytes()
    stream=b'q 1145.94 0 0 648 0 0 cm /Im Do Q\n'
    objs=[b'<< /Type /Catalog /Pages 2 0 R >>',
          b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
          b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1145.94 648] /Resources << /XObject << /Im 4 0 R >> >> /Contents 5 0 R >>',
          (f'<< /Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {len(data)} >>\nstream\n').encode()+data+b'\nendstream',
          f'<< /Length {len(stream)} >>\nstream\n'.encode()+stream+b'endstream']
    out=bytearray(b'%PDF-1.4\n'); offsets=[0]
    for i,obj in enumerate(objs,1):
        offsets.append(len(out));out+=f'{i} 0 obj\n'.encode()+obj+b'\nendobj\n'
    start=len(out);out+=f'xref\n0 {len(offsets)}\n0000000000 65535 f \n'.encode()
    for offset in offsets[1:]:out+=f'{offset:010d} 00000 n \n'.encode()
    out+=f'trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n'.encode()
    dest.write_bytes(out)


def pptx_from_base(base: Path, dest: Path, pngs: dict[int,Path]):
    with zipfile.ZipFile(base) as z: data={name:z.read(name) for name in z.namelist()}
    for n,path in pngs.items():data[f'ppt/media/image{n}.png']=path.read_bytes()
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for name,body in data.items():z.writestr(name,body)


def build_site():
    modules=[('概念地圖',1,4,'先弄清楚機器學習在做什麼。'),
             ('模型流程',5,13,'跟著同一個房價例子，完成資料到預測。'),
             ('動手預測',14,15,'自己算一次，再檢查答案與限制。')]
    nav=''.join(f'<a href="#module-{i}">{escape(name)}</a>' for i,(name,_,_,_) in enumerate(modules,1))
    blocks=[]
    for mi,(name,start,end,intro) in enumerate(modules,1):
        cards=[]
        for n in range(start,end+1):
            title,body,remember=CONTENT[n-1]
            if n == 15:
                cards.append(f'''<article class="lesson" id="lesson-15"><div class="number">15</div><div class="lesson-copy"><h3>{escape(title)}</h3><details><summary>完成練習後查看答案</summary><p>{escape(body)}</p><p class="remember"><strong>記住：</strong>{escape(remember)}</p><img src="./slides/slide-15.png" alt="答案與檢查的投影片" loading="lazy" width="2000" height="1125"></details></div></article>''')
            else:
                colab_link = '<p><a href="https://colab.research.google.com/" target="_blank" rel="noopener noreferrer">開啟 Google Colab ↗</a></p>' if n == 14 else ''
                cards.append(f'''<article class="lesson" id="lesson-{n}"><div class="number">{n:02d}</div><div class="lesson-copy"><h3>{escape(title)}</h3><p>{escape(body)}</p><p class="remember"><strong>記住：</strong>{escape(remember)}</p>{colab_link}<details><summary>查看第 {n} 頁投影片</summary><img src="./slides/slide-{n:02d}.png" alt="{escape(title)}的投影片" loading="lazy" width="2000" height="1125"></details></div></article>''')
        blocks.append(f'<section class="module" id="module-{mi}"><div class="module-head"><span>單元 {mi:02d}</span><h2>{escape(name)}</h2><p>{escape(intro)}</p></div>{"".join(cards)}</section>')
    html=f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><link rel="icon" href="./favicon.svg" type="image/svg+xml"><title>機器學習，從資料到預測｜學生講義</title><meta name="description" content="15 頁機器學習入門講義：概念、資料、訓練、測試與房價預測練習。"><style>
:root{{--ink:#18332f;--green:#176d59;--paper:#f7faf6;--line:#d5e2d9;--muted:#52645e}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:system-ui,-apple-system,"Noto Sans TC",sans-serif;line-height:1.7}}a{{color:var(--green)}}header,main,footer{{max-width:1050px;margin:auto;padding:24px}}header{{display:flex;justify-content:space-between;align-items:center;gap:20px;border-bottom:1px solid var(--line)}}.brand{{font-weight:750;text-decoration:none;color:var(--ink)}}nav{{display:flex;gap:16px;flex-wrap:wrap}}nav a{{text-decoration:none;font-weight:650}}.hero{{padding:64px 0 48px}}.eyebrow,.number,.module-head span{{font-size:.9rem;font-weight:750;color:var(--green);letter-spacing:.08em}}h1{{font-size:clamp(2.3rem,5vw,4rem);line-height:1.2;margin:12px 0}}.hero p{{font-size:1.16rem;max-width:700px;color:var(--muted)}}.actions{{display:flex;gap:12px;flex-wrap:wrap;margin-top:28px}}.button{{display:inline-block;padding:10px 18px;border-radius:9px;background:var(--green);color:white;text-decoration:none;font-weight:650}}.button.secondary{{background:white;color:var(--green);border:1px solid var(--line)}}.module{{margin:40px 0 68px}}.module-head{{border-top:3px solid var(--green);padding:18px 0 12px}}.module-head h2{{font-size:2rem;margin:4px 0}}.module-head p{{margin:0;color:var(--muted)}}.lesson{{display:grid;grid-template-columns:64px 1fr;gap:20px;border-top:1px solid var(--line);padding:27px 0}}.lesson h3{{font-size:1.32rem;margin:0 0 7px}}.lesson p{{margin:7px 0}}.remember{{color:var(--green)}}details{{margin-top:13px}}summary{{cursor:pointer;font-weight:650;color:var(--green)}}details img{{display:block;width:100%;height:auto;margin-top:13px;border-radius:8px}}.resources{{padding:28px;background:#eaf3ec;border-radius:12px}}.resources h2{{margin-top:0}}footer{{border-top:1px solid var(--line);color:var(--muted);font-size:.92rem}}@media(max-width:600px){{header{{display:block}}nav{{margin-top:12px}}.lesson{{grid-template-columns:42px 1fr;gap:10px}}}}@media print{{header nav,.actions,details,.resources{{display:none}}.lesson{{break-inside:avoid}}}}
</style></head><body><header><a class="brand" href="#top">徐國堂 Python 實作應用班</a><nav aria-label="講義目錄">{nav}<a href="./glossary/">延伸互動教材</a></nav></header><main id="top"><section class="hero"><div class="eyebrow">學生講義 · 15 頁</div><h1>機器學習，從資料到預測</h1><p>用同一個房價例子，依序弄懂概念、資料、訓練、測試與預測。每一頁只回答一個問題。</p><div class="actions"><a class="button" href="#module-1">開始閱讀</a><a class="button secondary" href="./downloads/機器學習簡報.pdf" download>下載 PDF</a><a class="button secondary" href="./downloads/機器學習簡報.pptx" download>下載 PowerPoint</a></div></section>{"".join(blocks)}<section class="resources"><h2>練完後還想繼續？</h2><p>到 <a href="./glossary/">名詞解釋互動教材</a> 操作資料分割、特徵工程與評估動畫。主講義先掌握流程即可。</p></section></main><footer>徐國堂 Python 實作應用班 · 機器學習學生講義</footer></body></html>'''
    html = html.replace('<a href="./glossary/">延伸互動教材</a></nav>',
                        '<a href="./projects/">實戰專案</a><a href="./glossary/">延伸互動教材</a></nav>')
    html = html.replace('<h2>練完後還想繼續？</h2>',
                        '<h2>練完後還想繼續？</h2><p>到 <a href="./projects/">實戰專案目錄</a> 選一個迴歸、分類或分群題目。</p>')
    (DOCS/'index.html').write_text(html,encoding='utf-8')


def main():
    SOURCE.mkdir(exist_ok=True)
    for stale in (SOURCE/'slide-16.svg',DOCS/'slides'/'slide-16.png'):
        if stale.exists():stale.unlink()
    with tempfile.TemporaryDirectory(prefix='ml-handout-') as tmpstr:
        tmp=Path(tmpstr);pngs={}
        for n,creator in enumerate(SLIDES,1):
            source=SOURCE/f'slide-{n:02d}.svg'
            source.write_text(svg(creator()),encoding='utf-8')
            png=tmp/f'slide-{n:02d}.png'
            run('rsvg-convert','-w','2000','-h','1125','-o',str(png),str(source))
            pngs[n]=png
            shutil.copyfile(png,DOCS/'slides'/f'slide-{n:02d}.png')
        pages=[]
        for n in range(1,16):
            jpeg=tmp/f'page-{n:02d}.jpg';pdf=tmp/f'page-{n:02d}.pdf'
            run('sips','-s','format','jpeg','-s','formatOptions','88',str(pngs[n]),'--out',str(jpeg))
            size=subprocess.check_output(['sips','-g','pixelWidth','-g','pixelHeight',str(jpeg)],text=True)
            width=int(re.search(r'pixelWidth: (\d+)',size).group(1));height=int(re.search(r'pixelHeight: (\d+)',size).group(1))
            jpeg_pdf(jpeg,pdf,width,height);pages.append(str(pdf))
        output_pdf=tmp/'handout.pdf';run('pdfunite',*pages,str(output_pdf))
        output_pptx=tmp/'handout.pptx';pptx_from_base(SOURCE/'base.pptx',output_pptx,pngs)
        for folder in (ROOT,DOCS/'downloads'):
            shutil.copyfile(output_pdf,folder/'機器學習簡報.pdf')
            shutil.copyfile(output_pptx,folder/'機器學習簡報.pptx')
    build_site()


if __name__=='__main__':main()
