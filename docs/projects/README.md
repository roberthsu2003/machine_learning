# 實戰專案導覽維護說明

[網站目錄](./index.html)與 22 個專案 README 頂端的「從這裡開始」由根目錄的 [build_project_guides.py](../../build_project_guides.py) 產生。

新增或修改專案時，在腳本的 `PROJECTS` 清單設定資料夾、任務類型、程度、學生要回答的問題與完成檢查，再執行：

```bash
python3 build_project_guides.py
```

腳本會保留各專案 README 既有的詳細內容，放在可展開區塊。Notebook、CSV 與 `app.py` 由實際資料夾檢查；缺少檔案時會停止，不會產生失效連結。
