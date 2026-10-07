# Herdr 使用文件

適用版本：herdr 0.8.0

## 目錄
- [第 1 章：Herdr 是什麼](#第-1-章herdr-是什麼)
- [第 2 章：安裝與初始設定](#第-2-章安裝與初始設定)
- [第 3 章：核心概念](#第-3-章核心概念)
- [第 4 章：核心場景手冊](#第-4-章核心場景手冊)
- [第 5 章：多 Agent 協作 SOP](#第-5-章多-agent-協作-sop)
- [第 6 章：指令速查表](#第-6-章指令速查表)

## 第 1 章：Herdr 是什麼

Herdr 是一個終端機工作區管理器，專為 AI coding agent 協作而設計。它把終端機組織成 workspace / tab / pane 三層結構，並能辨識 pane 內執行的 coding agent，自動回報其生命週期狀態。

與一般 tmux / iTerm 分頁相比，Herdr 的差異在於能辨識 pane 內的 coding agent 並回報其生命週期狀態，而非僅提供畫面分割。

四層概念：

| 層級 | 範例 ID | 說明 |
|---|---|---|
| workspace | w1 | 最上層工作區 |
| tab | w1:t1 | 工作區內的分頁 |
| pane | w1:p1 | 分頁內的終端機面板 |
| agent | claude | 運行於 pane 內的 coding agent |

## 第 2 章：安裝與初始設定

macOS 安裝：
```bash
brew install herdr
```

其他平台安裝方式：
請參考官方網站 https://herdr.dev 取得對應平台的安裝說明。

升級管道：
```bash
herdr update
herdr channel set stable
herdr channel set preview
```

zsh 補完：
```bash
herdr completion zsh
```

設定檔與記錄檔：
- 設定檔：`~/.config/herdr/config.toml`
- 記錄檔：`~/.config/herdr/herdr.log`、`herdr-client.log`、`herdr-server.log`
- 環境變數：`HERDR_CONFIG_PATH` 用於覆寫設定檔路徑

Agent 整合安裝：
```bash
herdr integration install claude
herdr integration install codex
herdr integration install opencode
```

驗證安裝：
```bash
herdr --version
herdr status
```
進入 session 後應有：
```bash
echo $HERDR_ENV
```
預期輸出為 `1`。

## 第 3 章：核心概念

三層結構與 ID 規則：
- workspace ID：`w1`
- tab ID：`w1:t1`
- pane ID：`w1:p1`
ID 為不可重用的穩定 handle。

注意：`pane move` 後會取得新的 workspace-qualified pane ID，舊 ID 失效。

環境變數注入：
- `HERDR_ENV`
- `HERDR_WORKSPACE_ID`
- `HERDR_TAB_ID`
- `HERDR_PANE_ID`

Agent 生命週期狀態：
- `idle`：等待輸入
- `working`：工作中
- `blocked`：等待使用者回應
- `done`：背景工作完成後的 idle，CLI 讀取不會標記為已看見
- `unknown`：狀態不明，不等於完成

`agent start` 只能在已存在且處於互動提示符的 shell pane 上啟動，不會自行建立或分割版面。

讀取來源 `--source` 四種模式：
- `visible`
- `recent`
- `recent-unwrapped`
- `detection`
日誌與逐字稿建議使用 `recent-unwrapped`。

## 第 4 章：核心場景手冊

### S1 開一個旁邊的 pane 跑測試／腳本，焦點不離開目前視窗

情境描述：
在不切換目前焦點的情況下，於右側新增 pane 執行測試腳本。

完整可複製指令：
```bash
herdr pane split --current --direction right --cwd "$PWD" --no-focus
```
從回應 `.result.pane.pane_id` 取 `<pane-id>`。

```bash
herdr pane run <pane-id> uv run pytest
```

```bash
herdr pane wait-output <pane-id> --match "passed" --timeout 60000
```

```bash
herdr pane read <pane-id> --source recent-unwrapped
```

預期輸出：
分割成功回傳 pane ID，wait-output 在測試完成時結束，read 取得完整輸出。

注意事項：
背景工作一律加 `--no-focus`，ID 需從 JSON 回應解析取得。

### S2 在新 pane 啟動另一個 coding agent 並派工、等待、讀取結果

情境描述：
為子 agent 建立獨立 pane，啟動後下達任務並等待完成。

完整可複製指令：
```bash
herdr pane split --current --direction down --cwd "$PWD" --no-focus
```
取得 `<pane-id>`。

```bash
herdr agent start sub-agent --kind claude --pane <pane-id>
```

```bash
herdr agent prompt sub-agent "請摘要本目錄結構" --wait --timeout 120000
```

```bash
herdr agent get sub-agent
herdr agent read sub-agent
```

預期輸出：
agent start 成功回報偵測，prompt 等待直至 `idle` / `done`，get / read 取得對話與輸出。

注意事項：
若 agent 已處於 `blocked`，`agent prompt` 會回傳 `agent_blocked`，需先 `herdr agent get` / `herdr agent read` 確認對話框，再用 `herdr agent send-keys sub-agent esc` 回應。

若 agent 在 alternate screen 執行，加大 `--lines` 仍讀不到完整回覆時，請 agent 將完整回覆寫成暫存目錄的 Markdown 檔並回傳路徑，再直接讀檔。

### S3 用 workspace 隔離不同任務

情境描述：
為不同任務建立獨立 workspace，避免互相干擾。

完整可複製指令：
```bash
herdr workspace list
```

```bash
herdr workspace create --label "course" --cwd "$PWD" --no-focus
```

```bash
herdr workspace get <workspace-id>
herdr workspace focus <workspace-id>
herdr workspace rename <workspace-id> "new-label"
```

> ⚠️ 危險操作：`herdr workspace close <workspace-id>` 會關閉整個工作區，請謹慎使用。

預期輸出：
list 顯示現有工作區，create 回傳 `workspace`、`tab`、`root_pane`。

注意事項：
新專案或隔離分支建議開新 workspace，同一任務內的排版只需在同 tab 分割 pane。

### S4 長時間跑 Playwright 或 dev server，離線後再回來

情境描述：
使用具名 session 讓長時間行程持續在背景。

完整可複製指令：
```bash
herdr --session devserver
```

```bash
herdr session list
```

```bash
herdr session attach devserver
```

```bash
herdr session stop devserver
```
> ⚠️ 危險操作：`herdr session stop <name>` 會停止該具名 session。

```bash
herdr session delete devserver
```
> ⚠️ 危險操作：`herdr session delete <name>` 會永久刪除已停止的 session。

預期輸出：
session list 顯示所有具名 session，attach 成功連回原工作區。

注意事項：
離線後重新 attach 可接續原 pane 及 agent 狀態。

## 第 5 章：多 Agent 協作 SOP

派工流程範例：
主控 agent 需要子 agent 撰寫一份文件，可依下列步驟執行。

```bash
herdr pane split --current --direction down --cwd "$PWD" --no-focus
```
取得 `<pane-id>`。

```bash
herdr agent start writer --kind claude --pane <pane-id>
```

```bash
herdr agent prompt writer "在 $PWD 撰寫一份 README_SUMMARY.md，摘要專案結構與使用方式" --wait --timeout 180000
```

```bash
herdr agent get writer
herdr agent read writer --source recent-unwrapped
```
驗收不通過時，附上具體缺失清單後退回重做：
```bash
herdr agent prompt writer "上一版缺少安裝步驟，請補上 brew install herdr 與 herdr update 範例" --wait --timeout 120000
```

驗收檢查清單範本：
- 指令是否可直接複製執行
- ID 是否使用佔位符
- 場景四段是否齊全
- 安全規則是否全部遵守

安全與禮儀規則：
- 背景工作一律加 `--no-focus`，除非主人要求切換畫面
- 一律使用 `--current`、明確 pane ID 或唯一 agent 名稱，不要依賴其他 client 的焦點 pane
- ID 必須從 JSON 回應解析，不可從側邊欄順序或範例推測
- 不關閉自己沒有建立的 workspace / tab / pane / session
- 絕不在活動 session 中執行 `herdr server stop`，絕不 kill 主 herdr 行程
- 探索指令請執行 `herdr <group>` 不帶子命令，不要執行裸 `herdr` 會啟動 TUI，也不要用省略參數試探會造成變更的巢狀命令
- 版面禮儀：寬的 pane 往右切、窄或高的 pane 往下切，避免連續同方向分割產生細長欄位，可先用 `herdr pane layout` 判斷

agent 命名規則：
符合 `[a-z][a-z0-9_-]{0,31}`，且在存活 agent 中唯一；結束後名稱會被清除。

## 第 6 章：指令速查表

### workspace
- `herdr workspace list` ｜ 列出工作區 ｜ 無 ｜ 唯讀
- `herdr workspace create` ｜ 建立工作區 ｜ `--cwd`、`--label`、`--no-focus` ｜ 會變更狀態
- `herdr workspace get` ｜ 取得工作區資訊 ｜ `<workspace-id>` ｜ 唯讀
- `herdr workspace focus` ｜ 切換焦點 ｜ `<workspace-id>` ｜ 會變更狀態
- `herdr workspace rename` ｜ 更名 ｜ `<workspace-id> <label>` ｜ 會變更狀態
- `herdr workspace close` ｜ 關閉工作區 ｜ 無 ｜ 會變更狀態

> ⚠️ 危險操作：`herdr workspace close <workspace-id>` 會關閉整個工作區。

### tab
- `herdr tab list` ｜ 列出分頁 ｜ 無 ｜ 唯讀
- `herdr tab create` ｜ 建立分頁 ｜ 無 ｜ 會變更狀態
- `herdr tab get` ｜ 取得分頁資訊 ｜ 無 ｜ 唯讀
- `herdr tab focus` ｜ 切換分頁 ｜ 無 ｜ 會變更狀態
- `herdr tab rename` ｜ 更名 ｜ 無 ｜ 會變更狀態
- `herdr tab close` ｜ 關閉分頁 ｜ 無 ｜ 會變更狀態

> ⚠️ 危險操作：`herdr tab close <tab-id>` 會關閉分頁。

### pane
- `herdr pane list` ｜ 列出面板 ｜ 無 ｜ 唯讀
- `herdr pane current` ｜ 取得目前面板 ｜ 無 ｜ 唯讀
- `herdr pane get` ｜ 取得面板資訊 ｜ 無 ｜ 唯讀
- `herdr pane layout` ｜ 查看版面 ｜ 無 ｜ 唯讀
- `herdr pane neighbor` ｜ 取得鄰近面板 ｜ `--direction`必填、`--pane`/`--current` ｜ 唯讀
- `herdr pane zoom` ｜ 切換放大 ｜ `--toggle`/`--on`/`--off`、 `--pane`/`--current` ｜ 會變更狀態
- `herdr pane move` ｜ 移動面板 ｜ `<pane-id>`、 `--tab + --split`、`--new-tab`、`--new-workspace` ｜ 會變更狀態
- `herdr pane split` ｜ 分割面板 ｜ `--current`、`--direction`、`--cwd`、`--no-focus` ｜ 會變更狀態
- `herdr pane run` ｜ 在面板執行指令 ｜ `<pane-id> <COMMAND>...` 位置參數 ｜ 會變更狀態
- `herdr pane read` ｜ 讀取輸出 ｜ `--source`、`--lines`、`--format` ｜ 唯讀
- `herdr pane wait-output` ｜ 等待輸出 ｜ `--match`、`--timeout`、`--source` ｜ 唯讀
- `herdr pane close` ｜ 關閉面板 ｜ 無 ｜ 會變更狀態

> ⚠️ 危險操作：`herdr pane close <pane-id>` 會關閉面板且 ID 不可重用。

### agent
- `herdr agent list` ｜ 列出 agent ｜ 無 ｜ 唯讀
- `herdr agent get` ｜ 取得 agent資訊 ｜ 無 ｜ 唯讀
- `herdr agent read` ｜ 讀取輸出 ｜ `--source`、`--lines`、`--format` ｜ 唯讀
- `herdr agent wait` ｜ 等待狀態 ｜ `--until`、`--timeout` ｜ 唯讀
- `herdr agent focus` ｜ 切換焦點 ｜ 無 ｜ 會變更狀態
- `herdr agent rename` ｜ 更名 ｜ 無 ｜ 會變更狀態
- `herdr agent start` ｜ 啟動 agent ｜ `--kind`、`--pane` ｜ 會變更狀態
- `herdr agent prompt` ｜ 提交提示 ｜ `--wait`、`--timeout`、`--until` ｜ 會變更狀態
- `herdr agent send-keys` ｜ 送鍵 ｜ `<TARGET> <KEY>...` 位置參數 ｜ 會變更狀態

### worktree
- `herdr worktree list` ｜ 列出 worktree ｜ 無 ｜ 唯讀
- `herdr worktree create` ｜ 建立 worktree ｜ `--branch`、`--base`、`--path`、`--label` ｜ 會變更狀態
- `herdr worktree open` ｜ 開啟 worktree ｜ `--path` / `--branch`（必填其一） ｜ 會變更狀態
- `herdr worktree remove` ｜ 移除 worktree ｜ `--workspace`必填、`--force` ｜ 會變更狀態

> ⚠️ 危險操作：`herdr worktree remove --force` 會強制移除工作樹。

### session
- `herdr session list` ｜ 列出 session ｜ 無 ｜ 唯讀
- `herdr session attach` ｜ 連接 session ｜ `<name>` ｜ 會變更狀態
- `herdr session stop` ｜ 停止 session ｜ `<name>` ｜ 會變更狀態
- `herdr session delete` ｜ 刪除 session ｜ `<name>` ｜ 會變更狀態

> ⚠️ 危險操作：`herdr session stop <name>` 會停止 session。
> ⚠️ 危險操作：`herdr session delete <name>` 會永久刪除已停止的 session。

### 其他
- `herdr integration install` ｜ 安裝整合 ｜ 無 ｜ 會變更狀態
- `herdr notification show` ｜ 顯示通知 ｜ `--body`、`--position`、`--sound` ｜ 會變更狀態
- `herdr status` ｜ 查看狀態 ｜ 無 ｜ 唯讀
- `herdr server stop` ｜ 停止伺服器 ｜ 無 ｜ 會變更狀態

> ⚠️ 危險操作：`herdr server stop` 會停止本機 herdr 伺服器。

JSON 回應欄位對照：
- `workspace create` → `.result.workspace` / `.result.tab` / `.result.root_pane`
- `tab create` → `.result.tab` / `.result.root_pane`
- `pane split` → `.result.pane.pane_id`
- `pane move` → `.result.move_result.pane.pane_id`（舊 ID 為 `.result.move_result.previous_pane_id`）
