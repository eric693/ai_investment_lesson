# 智投 AI｜AI 股票理專助理

> 善用 Claude AI，打造你的全自動投資分析系統

## 功能介紹

- 💬 **AI 對話理專**：用自然語言詢問任何投資問題
- 📊 **個股快速分析**：輸入股票代號，秒出基本面 + 技術面分析
- 📰 **財經新聞解讀**：貼上新聞，AI 自動整理投資影響
- 💼 **投資組合評估**：評估持股多元性與風險
- 🌐 **大盤展望**：一鍵生成今日市場分析
- 🔥 **快速詢問按鈕**：常用問題一鍵發問

## 本地執行

```bash
# 安裝依賴
npm install

# 設定環境變數
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env

# 啟動
npm start
```

開啟瀏覽器：http://localhost:3000

## 部署到 Render

### 方法一：使用 render.yaml（推薦）

1. 將專案推送到 GitHub
2. 前往 [render.com](https://render.com)，點擊 **New → Blueprint**
3. 選擇你的 GitHub repo
4. Render 會自動讀取 `render.yaml` 設定
5. 在環境變數中填入 `ANTHROPIC_API_KEY`
6. 點擊 **Apply** 完成部署

### 方法二：手動建立 Web Service

1. Render → New → **Web Service**
2. 連接 GitHub repo
3. 設定：
   - **Environment**: Node
   - **Build Command**: `npm install`
   - **Start Command**: `npm start`
4. 環境變數：
   - `ANTHROPIC_API_KEY` = `你的 Anthropic API Key`
5. 點擊 **Create Web Service**

## 取得 Anthropic API Key

前往 [console.anthropic.com](https://console.anthropic.com) 申請 API Key

## 免責聲明

本工具所有 AI 分析內容**僅供參考**，不構成任何投資建議。投資有風險，請自行評估後做出決策。
