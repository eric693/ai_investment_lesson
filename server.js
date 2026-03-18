const express = require('express');
const cors = require('cors');
const Anthropic = require('@anthropic-ai/sdk');
const OpenAI = require('openai');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ── API Clients ────────────────────────────────────
const anthropic = process.env.ANTHROPIC_API_KEY
  ? new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })
  : null;

const openai = process.env.OPENAI_API_KEY
  ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
  : null;

const SYSTEM = `你是一位專業的 AI 股票理財顧問「智投 AI」，專精台股與全球市場。
請用繁體中文回覆，條理清晰，善用表格與條列式。數據具體，語氣專業但親切。
所有分析僅供參考，不構成投資建議，投資人應自行評估風險。`;

// ── Which providers are available ─────────────────
app.get('/api/models', (req, res) => {
  res.json({
    claude: !!anthropic,
    openai: !!openai,
    default: anthropic ? 'claude' : (openai ? 'openai' : null),
  });
});

// ── Single call (Claude or OpenAI) ────────────────
async function ask(prompt, provider = 'claude', maxTokens = 2048) {
  if (provider === 'openai') {
    if (!openai) throw new Error('OpenAI API key 未設定');
    const r = await openai.chat.completions.create({
      model: 'gpt-4o', max_tokens: maxTokens,
      messages: [{ role: 'system', content: SYSTEM }, { role: 'user', content: prompt }],
    });
    return r.choices[0].message.content;
  } else {
    if (!anthropic) throw new Error('Anthropic API key 未設定');
    const r = await anthropic.messages.create({
      model: 'claude-opus-4-5', max_tokens: maxTokens, system: SYSTEM,
      messages: [{ role: 'user', content: prompt }],
    });
    return r.content[0].text;
  }
}

// ── Streaming chat ─────────────────────────────────
async function stream(messages, provider = 'claude', res) {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  if (provider === 'openai') {
    if (!openai) throw new Error('OpenAI API key 未設定');
    const s = await openai.chat.completions.create({
      model: 'gpt-4o', max_tokens: 2048, stream: true,
      messages: [{ role: 'system', content: SYSTEM }, ...messages],
    });
    for await (const chunk of s) {
      const text = chunk.choices[0]?.delta?.content || '';
      if (text) res.write(`data: ${JSON.stringify({ text })}\n\n`);
    }
  } else {
    if (!anthropic) throw new Error('Anthropic API key 未設定');
    const s = await anthropic.messages.stream({
      model: 'claude-opus-4-5', max_tokens: 2048, system: SYSTEM, messages,
    });
    for await (const c of s) {
      if (c.type === 'content_block_delta' && c.delta.type === 'text_delta')
        res.write(`data: ${JSON.stringify({ text: c.delta.text })}\n\n`);
    }
  }
  res.write('data: [DONE]\n\n');
  res.end();
}

// ── 1. Streaming chat ──────────────────────────────
app.post('/api/chat', async (req, res) => {
  try { await stream(req.body.messages, req.body.provider || 'claude', res); }
  catch (e) {
    console.error(e);
    res.write(`data: ${JSON.stringify({ text: `\n\n⚠️ 錯誤：${e.message}` })}\n\n`);
    res.write('data: [DONE]\n\n');
    res.end();
  }
});

// ── 2. Quick analysis (right panel) ───────────────
app.post('/api/quick-analysis', async (req, res) => {
  const { type, content, provider } = req.body;
  const prompts = {
    stock: `請全面分析股票 ${content}：\n1.公司簡介\n2.近期股價表現\n3.基本面指標（EPS/本益比/殖利率）\n4.技術面重點\n5.主要風險\n6.投資評估與目標價`,
    news: `請分析以下財經新聞的投資影響：\n${content}\n\n1.新聞摘要\n2.受影響產業/類股\n3.短期影響\n4.長期影響\n5.建議對策`,
    portfolio: `請評估以下投資組合：\n${content}\n\n1.多元性分析\n2.風險集中度\n3.預期報酬\n4.潛在風險\n5.優化建議`,
    market: `請分析今日台股市場狀況：\n1.大盤走勢\n2.強勢族群\n3.弱勢族群\n4.重要籌碼動向\n5.明日操作重點\n6.風險提示`,
  };
  try { res.json({ result: await ask(prompts[type] || content, provider) }); }
  catch (e) { res.status(500).json({ error: e.message || '分析失敗' }); }
});

// ── 3. Valuation module ────────────────────────────
app.post('/api/valuation', async (req, res) => {
  const { stock, price, pe, pb, roe, eps, growth, sector, provider } = req.body;
  const prompt = `請對「${stock}」進行完整估值分析：

目前股價：${price || '未提供'}
本益比(P/E)：${pe || '未提供'}
股價淨值比(P/B)：${pb || '未提供'}
股東權益報酬率(ROE)：${roe || '未提供'}%
每股盈餘(EPS)：${eps || '未提供'}
盈餘成長率：${growth || '未提供'}%
所屬產業：${sector || '未提供'}

請提供：
1. 估值方法分析（DDM / DCF / 相對估值法）
2. 合理價格區間（低估/合理/高估分界）
3. 與同業比較
4. 安全邊際評估
5. 買入/持有/賣出建議
6. 風險因素`;
  try { res.json({ result: await ask(prompt, provider, 3000) }); }
  catch (e) { res.status(500).json({ error: e.message || '估值分析失敗' }); }
});

// ── 4. Technical analysis ──────────────────────────
app.post('/api/technical', async (req, res) => {
  const { stock, price, ma5, ma20, ma60, kd_k, kd_d, rsi, macd, volume, trend, provider } = req.body;
  const prompt = `請對「${stock}」進行技術指標分析：

目前股價：${price}
MA5/MA20/MA60：${ma5} / ${ma20} / ${ma60}
KD指標 K值/D值：${kd_k} / ${kd_d}
RSI：${rsi}
MACD：${macd || '未提供'}
成交量趨勢：${volume || '未提供'}
近期走勢：${trend || '未提供'}

請分析：
1. 均線多空排列判斷
2. KD指標解讀（超買/超賣/黃金交叉/死亡交叉）
3. RSI強弱評估
4. 支撐與壓力位
5. 短中長期趨勢判斷
6. 操作策略建議（進場點/停損點/目標價）`;
  try { res.json({ result: await ask(prompt, provider, 2500) }); }
  catch (e) { res.status(500).json({ error: e.message || '技術分析失敗' }); }
});

// ── 5. Backtest analysis ───────────────────────────
app.post('/api/backtest', async (req, res) => {
  const { strategy, stock, period, capital, buyRule, sellRule, provider } = req.body;
  const prompt = `請對以下投資策略進行回測分析模擬：

標的：${stock || '台股大盤/ETF'}
策略名稱：${strategy}
回測期間：${period}
初始資金：${capital || '100萬'}
買入條件：${buyRule}
賣出條件：${sellRule}

請提供：
1. 策略邏輯評估（合理性分析）
2. 歷史回測模擬結果（年化報酬率、最大回撤、勝率估計）
3. 策略優點分析
4. 策略缺點與風險
5. 改進建議
6. 與大盤買進持有比較
7. 適合的市場環境

注意：此為 AI 模擬分析，非實際歷史數據回測`;
  try { res.json({ result: await ask(prompt, provider, 3000) }); }
  catch (e) { res.status(500).json({ error: e.message || '回測分析失敗' }); }
});

// ── 6. Portfolio allocation ────────────────────────
app.post('/api/allocation', async (req, res) => {
  const { holdings, risk, goal, horizon, provider } = req.body;
  const prompt = `請分析並優化以下投資組合配置：

目前持股：
${holdings}

風險承受度：${risk || '中等'}
投資目標：${goal || '穩健成長'}
投資期間：${horizon || '3-5年'}

請提供：
1. 當前組合健診（集中度/多元性/風險評分）
2. 產業/地區配置分析
3. 建議的最佳化配置比例
4. 需要減碼/增碼的標的說明
5. 可考慮加入的新標的（ETF/個股）
6. 預期風險收益比改善幅度`;
  try { res.json({ result: await ask(prompt, provider, 3000) }); }
  catch (e) { res.status(500).json({ error: e.message || '配置分析失敗' }); }
});

// ── 7. Investment report ───────────────────────────
app.post('/api/report', async (req, res) => {
  const { portfolio, period, focus, provider } = req.body;
  const prompt = `請為以下投資組合生成「${period || '本季'}投資分析報告」：

持股明細：
${portfolio}

重點關注：${focus || '整體績效與風險'}

報告格式：
# 投資分析報告 - ${period || '本季'}

## 一、執行摘要
## 二、市場環境回顧
## 三、持股績效分析
## 四、風險指標評估
## 五、重要事件影響
## 六、下季展望與策略
## 七、調整建議
## 八、免責聲明

請生成完整專業報告。`;
  try { res.json({ result: await ask(prompt, provider, 4096) }); }
  catch (e) { res.status(500).json({ error: e.message || '報告生成失敗' }); }
});

// ── 8. Monthly summary ─────────────────────────────
app.post('/api/monthly', async (req, res) => {
  const { month, provider } = req.body;
  const prompt = `請生成「${month || '本月'}台股市場重點摘要推播」：

格式要求：
# ${month || '本月'}投資摘要

## 大盤表現
## 產業亮點（前3強勢/前3弱勢）
## 重要財經事件
## 籌碼動向（外資/投信/自營商）
## 本月最強個股
## 下月關鍵觀察指標
## AI 投資建議重點
## 風險警示

請生成完整、有深度的每月投資摘要。`;
  try { res.json({ result: await ask(prompt, provider, 3000) }); }
  catch (e) { res.status(500).json({ error: e.message || '摘要生成失敗' }); }
});

// ── 9. Financial report prompt generator ──────────
app.post('/api/fin-prompt', async (req, res) => {
  const { stock, reportType, provider } = req.body;
  const prompt = `請生成一個用於分析「${stock}」${reportType || '財報'}的專業 AI 提問範本。

範本需要包含：
1. 財務健康度檢查清單
2. 關鍵指標分析框架
3. 與同業比較的問題
4. 風險評估問題
5. 未來展望評估

請提供完整的提問範本，讓使用者可以直接複製使用。`;
  try { res.json({ result: await ask(prompt, provider, 2000) }); }
  catch (e) { res.status(500).json({ error: e.message || '範本生成失敗' }); }
});

app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));

app.listen(PORT, () => console.log(`智投 AI 已啟動 Port:${PORT}`));
