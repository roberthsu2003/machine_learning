"""Build all static glossary pages from the existing chapter Markdown.

Run from repository root: .venv/bin/python docs/glossary/build.py
The browser needs no Python, package download, CDN, or Markdown runtime.
"""
from pathlib import Path
from html import escape
from urllib.parse import quote
import json
import re
import shutil
import mistune
from mistune.plugins.math import math_in_list, math_in_quote

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
MANIFEST = BASE / 'chapters.json'
catalog = json.loads(MANIFEST.read_text())
chapters = catalog['chapters']
github = 'https://github.com/roberthsu2003/machine_learning/blob/main/'


class Tex:
    """Native MathML for the finite TeX notation used by these 12 chapters."""
    symbols = dict(alpha='α', theta='θ', lambda_='λ', mu='μ', sigma='σ', nabla='∇',
                   times='×', cdot='·', approx='≈', pm='±', mid='|', in_='∈',
                   leftarrow='←', rightarrow='→', sum='∑', cdots='⋯', vdots='⋮',
                   ddots='⋱', lceil='⌈', rceil='⌉', chi='χ')

    def __init__(self, source):
        self.s = source
        self.i = 0

    def space(self):
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def raw_group(self):
        self.space()
        if self.i >= len(self.s) or self.s[self.i] != '{':
            return ''
        self.i += 1
        start, depth = self.i, 1
        while self.i < len(self.s) and depth:
            if self.s[self.i] == '{':
                depth += 1
            elif self.s[self.i] == '}':
                depth -= 1
            self.i += 1
        return self.s[start:self.i - 1]

    def expression(self, stop=False):
        result = []
        while self.i < len(self.s):
            self.space()
            if self.i >= len(self.s):
                break
            if self.s[self.i] == '}':
                if stop:
                    self.i += 1
                    break
                self.i += 1
                continue
            atom = self.atom()
            sub = sup = None
            while True:
                self.space()
                if self.i < len(self.s) and self.s[self.i] in '_^':
                    kind = self.s[self.i]
                    self.i += 1
                    argument = self.atom()
                    if kind == '_':
                        sub = argument
                    else:
                        sup = argument
                else:
                    break
            if sub and sup:
                atom = f'<msubsup>{atom}{sub}{sup}</msubsup>'
            elif sub:
                atom = f'<msub>{atom}{sub}</msub>'
            elif sup:
                atom = f'<msup>{atom}{sup}</msup>'
            result.append(atom)
        return '<mrow>' + ''.join(result) + '</mrow>'

    def atom(self):
        self.space()
        if self.i >= len(self.s):
            return '<mrow></mrow>'
        ch = self.s[self.i]
        self.i += 1
        if ch == '{':
            return self.expression(True)
        if ch == '\\':
            match = re.match(r'[A-Za-z]+', self.s[self.i:])
            if not match:
                if self.i < len(self.s):
                    ch = self.s[self.i]
                    self.i += 1
                return '<mo>' + escape(ch) + '</mo>'
            cmd = match.group()
            self.i += len(cmd)
            if cmd in ('left', 'right'):
                return self.atom()
            if cmd == 'frac':
                return '<mfrac>' + self.atom() + self.atom() + '</mfrac>'
            if cmd == 'sqrt':
                return '<msqrt>' + self.atom() + '</msqrt>'
            if cmd in ('hat', 'bar'):
                return '<mover accent="true">' + self.atom() + ('<mo>^</mo>' if cmd == 'hat' else '<mo>¯</mo>') + '</mover>'
            if cmd in ('text', 'mathbb'):
                value = escape(self.raw_group())
                return f'<mtext>{value}</mtext>' if cmd == 'text' else f'<mi mathvariant="double-struck">{value}</mi>'
            if cmd == 'begin':
                environment = self.raw_group()
                end = '\\end{' + environment + '}'
                found = self.s.find(end, self.i)
                if found >= 0:
                    content = self.s[self.i:found]
                    self.i = found + len(end)
                    rows = re.split(r'\\\\', content)
                    table = '<mtable>' + ''.join('<mtr>' + ''.join('<mtd>' + Tex(cell).expression() + '</mtd>' for cell in row.split('&')) + '</mtr>' for row in rows if row.strip()) + '</mtable>'
                    return '<mrow><mo>[</mo>' + table + '<mo>]</mo></mrow>'
            if cmd == 'quad':
                return '<mspace width="1em"/>'
            if cmd in ('min', 'max'):
                return f'<mo>{cmd}</mo>'
            symbol = self.symbols.get(cmd, self.symbols.get(cmd + '_', cmd))
            tag = 'mi' if cmd in ('alpha', 'theta', 'lambda', 'mu', 'sigma', 'chi') else 'mo'
            return f'<{tag}>{escape(symbol)}</{tag}>'
        if ch.isdigit():
            match = re.match(r'[\d.]*', self.s[self.i:])
            number = ch + match.group()
            self.i += len(match.group())
            return f'<mn>{number}</mn>'
        return f'<{"mi" if ch.isalpha() else "mo"}>{escape(ch)}</{"mi" if ch.isalpha() else "mo"}>'


def mathml(source, display=False):
    return f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="{"block" if display else "inline"}" aria-label="{escape(source, quote=True)}">{Tex(source).expression()}</math>'


def scene_html(scene):
    sid = scene['id']
    return f'''<section class="scene" id="{sid}" data-scene="{sid}" aria-labelledby="title-{sid}">
<div class="scene-head"><h3 id="title-{sid}">{escape(scene['title'])}</h3><span class="scene-tag">2D GEOMETRY · INTERACTIVE</span></div>
<p class="scene-description">{escape(scene['geometry'])}</p>
<canvas width="760" height="450" role="img" aria-label="{escape(scene['title'])}幾何動畫" aria-describedby="result-{sid}">請閱讀下方的動畫說明與本節教材。</canvas>
<div class="scene-parameters"></div>
<p class="scene-result" id="result-{sid}" role="status" aria-live="polite">按播放或單步，觀察本概念。</p>
<div class="scene-controls"><button data-action="play" aria-pressed="false">▶ 播放動畫</button><button data-action="reset">↺ 重設</button><button data-action="step">下一步 →</button><button data-action="resample" hidden>重抽資料</button><button data-action="zoom">放大動畫</button><label>速度<select class="speed" aria-label="{escape(scene['title'])}播放速度"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><input class="timeline" type="range" min="0" max="16" step="0.01" value="0" aria-label="{escape(scene['title'])}動畫時間軸"><span class="time-label">0%</span></div>
<p class="scene-note">動畫預設暫停，離開畫面自動暫停。拖動時間軸可回看；放大後按 Esc 可返回。</p>
</section>'''


class Renderer(mistune.HTMLRenderer):
    def __init__(self, chapter):
        super().__init__(escape=False)
        self.chapter = chapter
        self.source = ROOT / chapter['source_readme']
        self.used = set()
        self.placeholders = {}
        self.headings = []

    def image(self, text, url, title=None):
        path = (self.source.parent / url).resolve()
        relative = str(path.relative_to(ROOT))
        target = BASE / self.chapter['slug'] / 'assets/images' / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        matched = [scene for scene in self.chapter['scenes'] if relative in scene['source_images'] and scene['id'] not in self.used]
        for scene in matched:
            self.used.add(scene['id'])
        content = ''.join(scene_html(scene) for scene in matched)
        jump = ''.join(f'<a href="#{scene["id"]}">↑ 回到互動動畫：{escape(scene["title"])}</a>' for scene in matched)
        content += f'<details class="original"><summary>對照原教材圖解：{text}</summary>' + (f'<p class="scene-jump">{jump}</p>' if jump else '') + f'<img src="./assets/images/{quote(path.name)}" alt="{escape(text, quote=True)}" loading="lazy"><a href="./assets/images/{quote(path.name)}" target="_blank" rel="noopener">開啟完整尺寸原圖 ↗</a></details>'
        token = f'@@IMAGE{len(self.placeholders)}@@'
        self.placeholders[token] = content
        return token

    def link(self, text, url, title=None):
        if not url.startswith(('https:', 'http:', '#', 'mailto:')):
            path = (self.source.parent / url).resolve()
            matched = next((c for c in chapters if ROOT / c['source_readme'] == path), None)
            if matched:
                url = '../' + matched['slug'] + '/'
            elif path == ROOT / '名詞解釋/README.md':
                url = '../'
            else:
                url = github + quote(str(path.relative_to(ROOT)))
        return super().link(text, url, title)

    def heading(self, text, level, **attrs):
        if level == 1:
            return ''
        sid = f'section-{len(self.headings) + 1}'
        if level == 2:
            self.headings.append((sid, re.sub(r'<[^>]+>', '', text)))
        return f'<h{level} id="{sid}">{text}</h{level}>\n' if level == 2 else f'<h{level}>{text}</h{level}>\n'

    def block_code(self, code, info=None):
        if info == 'mermaid':
            labels = re.findall(r'\["([^"\]]+)"\]', code)
            labels = list(dict.fromkeys(labels))
            if labels:
                return '<div class="flow-summary"><strong>流程與概念摘要</strong><ul>' + ''.join('<li>' + escape(label).replace('&lt;br&gt;', ' ／ ') + '</li>' for label in labels) + '</ul></div>'
        return super().block_code(code, info)


def normalize(source):
    source = re.sub(r'\n\[⏮️ 上一章[^\n]*', '', source)
    source = re.sub(r'> \[!(?:TIP|CAUTION|WARNING|NOTE)\]\s*\n', '> ', source)
    source = re.sub(r'\$\$([\s\S]*?)\$\$', lambda m: '\n$$\n' + m.group(1).strip() + '\n$$\n' if '\n' in m.group(1) else m.group(), source)
    # Keep the source structure while correcting a few categorical overstatements.
    changes = {
        '與原始數據集完全一致': '盡量接近原始數據集（受整數樣本數限制）',
        '皆精準維持 2%': '盡量維持約 2%（受樣本數限制）',
        '介於 0.5 到 1.0 之間': '介於 0 到 1 之間；低於 0.5 可能表示排序方向反轉',
        '完全不受類別不平衡干擾': '衡量跨門檻的排序能力，但類別不平衡時仍應搭配 PR 指標與情境分析',
        '**較不敏感 (Robust)**': '**仍會受離群值影響**',
        '當前所有 AI 均屬於此類': '此處用特定任務系統作為範例',
        '表現無上限': '表現仍受資料、架構與訓練條件限制',
        '說明模型架構設計存在重大錯誤': '可能與模型、資料品質或分布改變有關，需進一步診斷',
    }
    for old, new in changes.items():
        source = source.replace(old, new)
    return source


def navigation(chapter):
    i = chapter['number'] - 1
    previous = f'<a href="../{chapters[i-1]["slug"]}/">← {i:02} {chapters[i-1]["title"]}</a>' if i else '<span>第一章</span>'
    following = f'<a href="../{chapters[i+1]["slug"]}/">{i+2:02} {chapters[i+1]["title"]} →</a>' if i < 11 else '<span>已到最後一章</span>'
    return f'<nav class="chapter-nav" aria-label="章節導覽">{previous}<a href="../">返回 12 章目錄</a>{following}</nav>'


for chapter in chapters:
    renderer = Renderer(chapter)
    md = mistune.create_markdown(renderer=renderer, plugins=['table', 'math', 'task_lists', math_in_list, math_in_quote])
    renderer.register('inline_math', lambda renderer, text: mathml(text))
    renderer.register('block_math', lambda renderer, text: '<div class="math-block">' + mathml(text, True) + '</div>')
    source = normalize((ROOT / chapter['source_readme']).read_text())
    # On subsequent builds, don't duplicate the generated entry in the webpage.
    source = re.sub(r'<!-- interactive-glossary:start -->[\s\S]*?<!-- interactive-glossary:end -->', '', source)
    body = md(source)
    for token, content in renderer.placeholders.items():
        body = body.replace('<p>' + token + '</p>', content).replace(token, content)
    extra = ''.join(scene_html(scene) for scene in chapter['scenes'] if scene['id'] not in renderer.used)
    if extra:
        summary = re.search(r'<h2[^>]*>[^<]*📌', body)
        body = body[:summary.start()] + extra + body[summary.start():] if summary else body + extra
    body = re.sub(r'(<table>[\s\S]*?</table>)', r'<div class="table-scroll">\1</div>', body)
    # A displayed formula can also appear in a table cell; never nest a div in a paragraph.
    body = re.sub(r'<p>(<div class="math-block">[\s\S]*?</div>)</p>', r'\1', body)
    title = chapter['title']
    toc = ''.join(f'<a href="#{scene["id"]}">{escape(scene["title"])}</a>' for scene in chapter['scenes'])
    toc += '<hr>' + ''.join(f'<a href="#{sid}">{escape(name)}</a>' for sid, name in renderer.headings)
    source_url = github + quote(chapter['source_readme'])
    html = f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><meta name="description" content="第 {chapter['number']:02} 章 {escape(title)}：完整教材、原圖對照與互動 2D 幾何動畫。"><title>{chapter['number']:02} {escape(title)}｜機器學習互動教材</title><link rel="icon" href="../../favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="../shared/lesson.css"><script type="module" src="./lesson.js"></script></head><body>
<header class="site-header"><a class="brand" href="../">機器學習 ／ 名詞解釋</a><nav aria-label="主選單"><a href="../">12 章目錄</a><a href="../../">簡報教材</a><a href="{source_url}">GitHub 原文 ↗</a></nav></header>
<main class="page-wrap"><section class="chapter-hero"><div class="eyebrow">CHAPTER {chapter['number']:02} / 12 · INTERACTIVE LESSON</div><h1>{escape(title)}</h1><p>沿用原教材順序，用幾何圖形的移動與變化理解本章概念。點選播放，或調整參數觀察結果。</p><div class="hero-actions"><a class="button primary" href="#{chapter['scenes'][0]['id']}">開始本章動畫 ↓</a><button class="button" data-print>列印本章教材</button></div></section>
<noscript><p class="no-js-note">JavaScript 未啟用，動畫無法播放；完整文字、公式、表格與原圖仍可閱讀。</p></noscript>
<div class="lesson-layout"><aside class="lesson-toc" aria-label="本章目錄"><h2>動畫與教材</h2><nav class="toc-links">{toc}</nav></aside><article class="lesson-content">{body}<p class="source-note">教材來源：<a href="{source_url}">原章節 README</a>。动画示例的假設與計算方式，標示於各場景下方。</p>{navigation(chapter)}</article></div></main><footer>徐國堂 Python 實作應用班 · 機器學習互動教材 · Light Mode</footer></body></html>'''
    html = html.replace('动画', '動畫')
    chapter_dir = BASE / chapter['slug']
    (chapter_dir / 'index.html').write_text(html)
    (chapter_dir / 'lesson.js').write_text("import {mountScenes} from '../shared/player.js';\nimport {definitions} from './scenes/index.js';\nmountScenes(definitions);\n")
    chapter['status'] = 'implemented'

modules = [('模組一：數據與特徵基石', 0, 4), ('模組二：模型與優化機制', 4, 6), ('模組三：經典演算法與集成', 6, 8), ('模組四：評估診斷與實戰落地', 8, 12)]
sections = ''
for module, first, last in modules:
    cards = ''
    for chapter in chapters[first:last]:
        topics = '、'.join(scene['title'] for scene in chapter['scenes'])
        cards += f'<a class="chapter-card" href="./{chapter["slug"]}/"><small>CHAPTER {chapter["number"]:02}</small><h3>{chapter["title"]}</h3><p>{escape(topics)}</p><span>{len(chapter["scenes"])} 個互動場景 · 開始學習 →</span></a>'
    sections += f'<section class="module"><h2>{module}</h2><div class="chapter-grid">{cards}</div></section>'
(BASE / 'index.html').write_text(f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><meta name="description" content="12 章機器學習名詞解釋，42 個可操作的 2D 幾何動畫：資料、模型、演算法與評估。"><title>機器學習名詞解釋｜12 章互動動畫</title><link rel="icon" href="../favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="./shared/lesson.css"></head><body><header class="site-header"><a class="brand" href="./">機器學習 ／ 名詞解釋</a><nav><a href="../">簡報教材</a><a href="https://github.com/roberthsu2003/machine_learning/tree/main/名詞解釋">GitHub 教材 ↗</a></nav></header><main class="page-wrap"><section class="index-intro"><div class="eyebrow">12 CHAPTERS · 42 GEOMETRIC ANIMATIONS</div><h1>讓抽象名詞，<br>變成看得見的過程。</h1><p>依你熟悉的 01–12 章順序，從資料與特徵走到模型訓練。移動樣本、調整步長、比較模型，親手觀察每個概念如何運作。</p><a class="button primary" href="./01-learning-paradigms/">從第 01 章開始 →</a></section>{sections}</main><footer>徐國堂 Python 實作應用班 · 全部動畫在瀏覽器執行，無須登入或安裝套件。</footer></body></html>''')
catalog['status'] = 'implemented'
MANIFEST.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n')
print(f'Built {len(chapters)} chapter pages, index, and {sum(len(c["scenes"]) for c in chapters)} animation mounts.')
