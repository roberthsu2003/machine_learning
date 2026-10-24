# 教材安全與套件維護

這是教學儲存庫，沒有獨立的軟體發行版本。課堂環境以根目錄的 `pyproject.toml` 和 `uv.lock` 為準。

```bash
uv sync --locked
uv audit --frozen
```

GitHub Dependabot 每週檢查 `uv` 依賴更新。合併更新前，應重跑套件匯入檢查與相關 Notebook。

教材中的金鑰、權杖與密碼一律使用佔位文字或環境變數，不應寫入 Notebook 或提交至 Git。`LINE Notify` 已停止服務，相關 Notebook 只保留作為歷史範例。
