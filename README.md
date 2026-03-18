# 智投 AI｜全方位 AI 投資分析系統

> 整合 Yahoo Finance 即時股價 + Claude / GPT-4o，打造真實數據驅動的投資助理

## 功能一覽

| 模組 | 說明 |
|---|---|
| AI 對話理專 | Claude / GPT-4o 串流對話，支援切換 |
| 即時股價 Ticker | Yahoo Finance 自動更新，台股＋美股＋匯率＋黃金 |
| 自選股報價 | 5 支股票即時股價、漲跌幅，60 秒自動刷新 |
| 模組化提問範本 | 16 個精選投資分析範本，一鍵套用 |
| AI 投資儀表板 | 台積電歷史走勢圖（真實數據）、產業配置、指數 KPI |
| 市場情緒指標 | 外資買賣超、大盤漲跌（TWSE API）、VIX |
| AI 估值模組 | 輸入財務數據，AI 自動帶入即時股價分析 |
| 技術指標分析 | 帶入近 30 日歷史資料輔助 AI 判斷 |
| 回測分析模組 | 策略回測模擬 |
| 投資組合配置 | AI 分析多元性與風險 |
| 自動化投資報告 | 一鍵生成完整季報 |
| 每月重點摘要 | AI 生成月度市場摘要推播 |
| 使用說明 & Python 範例 | 帶入手工作流與程式碼範例 |

## 本地執行

```bash
npm install

# 建立 .env（至少填一個 AI Key）
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
echo "OPENAI_API_KEY=sk-..." >> .env

npm start
# 開啟 http://localhost:3000
```

## 部署到 Render

1. 將專案推上 GitHub
2. Render → **New → Web Service** → 選擇 repo
3. 設定：
   - Build Command：`npm install`
   - Start Command：`npm start`
4. 環境變數（至少一個）：
   - `ANTHROPIC_API_KEY`
   - `OPENAI_API_KEY`
5. 點 **Deploy** — 約 2 分鐘完成

## 資料來源

| 資料 | 來源 | 更新頻率 |
|---|---|---|
| 股價、指數、匯率、黃金 | Yahoo Finance（免費） | 60 秒 |
| 外資買賣超 | 證交所 TWSE 開放 API（免費） | 120 秒 |
| AI 分析 | Claude claude-opus-4-5 / GPT-4o | 即時 |

## 免責聲明

所有 AI 分析內容僅供參考，不構成任何投資建議。投資有風險，請自行評估。
