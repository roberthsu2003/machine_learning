# 機器學習學生講義網站

[index.html](./index.html) 是 15 頁講義的文字版，按「概念地圖 → 模型流程 → 動手預測」分成三個單元。每節先有白話重點，再可展開投影片。網站不需要後端或外部 JavaScript。

`slides/` 保存投影片預覽；`downloads/` 保存與[講義來源](../簡報圖片/README.md)相同版本的 PDF 和 PowerPoint。修改講義時，執行 `python3 簡報圖片/build_handout.py` 會同步更新這些檔案與網站。

[名詞解釋互動教材](./glossary/index.html)保留為延伸閱讀，不放進入門主線。GitHub Pages 可將 `docs/` 設為發布目錄。
