# 機器學習課程網頁

Light Mode 靜態網頁入口：[index.html](./index.html)。直接在瀏覽器開啟即可使用，無須安裝套件。

正式網址：<https://roberthsu2003.github.io/machine_learning/>

GitHub Pages 設定：在專案 **Settings → Pages → Build and deployment** 選擇 **Deploy from a branch**，分支選 **main**，目錄選 **/docs**。提交並推送後，等待 Pages 部署完成即可使用上述網址。

PDF 與 PowerPoint 下載檔案放在 `docs/downloads/`，會隨網頁一起發布。更新原始教材時，請同步更新此資料夾內的副本。

## 完整互動教材

網頁包含 15 章完整課程，對應原簡報的每一頁。可使用目錄跳轉、逐章教學模式與列印；每章可展開原始投影片。JavaScript 提供學習流程、貓狗分類損失更新、六筆房價資料的線性回歸與多特徵推論示範。

`course.css` 與 `course.js` 為本機靜態資源；`slides/` 保存由原 PowerPoint 擷取的 15 頁圖片。發布時請一併保留這些檔案。互動示範在瀏覽器執行，不需後端或外部 JavaScript 套件；新增數值模型皆在頁面標示為教學用途。
