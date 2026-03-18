const express   = require('express');
const cors      = require('cors');
const path      = require('path');
const Anthropic  = require('@anthropic-ai/sdk');
const OpenAI    = require('openai');
const yf        = require('yahoo-finance2').default;

const app  = express();
const PORT = process.env.PORT || 3000;
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const anthropic = process.env.ANTHROPIC_API_KEY ? new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY }) : null;
const openai    = process.env.OPENAI_API_KEY    ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY })    : null;

const SYSTEM = `你是一位專業的 AI 股票理財顧問「智投 AI」，專精台股與全球市場。
請用繁體中文回覆，條理清晰，善用表格與條列式。數據具體，語氣專業但親切。
所有分析僅供參考，不構成投資建議，投資人應自行評估風險。`;

app.get('/api/models', (_req, res) => {
  res.json({ claude: !!anthropic, openai: !!openai, default: anthropic ? 'claude' : (openai ? 'openai' : null) });
});

async function ask(prompt, provider = 'claude', maxTokens = 2048) {
  if (provider === 'openai') {
    if (!openai) throw new Error('OpenAI API key 未設定');
    const r = await openai.chat.completions.create({ model: 'gpt-4o', max_tokens: maxTokens, messages: [{ role: 'system', content: SYSTEM }, { role: 'user', content: prompt }] });
    return r.choices[0].message.content;
  }
  if (!anthropic) throw new Error('Anthropic API key 未設定');
  const r = await anthropic.messages.create({ model: 'claude-opus-4-5', max_tokens: maxTokens, system: SYSTEM, messages: [{ role: 'user', content: prompt }] });
  return r.content[0].text;
}

async function streamAI(messages, provider = 'claude', res) {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  if (provider === 'openai') {
    if (!openai) throw new Error('OpenAI API key 未設定');
    const s = await openai.chat.completions.create({ model: 'gpt-4o', max_tokens: 2048, stream: true, messages: [{ role: 'system', content: SYSTEM }, ...messages] });
    for await (const chunk of s) { const t = chunk.choices[0]?.delta?.content || ''; if (t) res.write(`data: ${JSON.stringify({ text: t })}\n\n`); }
  } else {
    if (!anthropic) throw new Error('Anthropic API key 未設定');
    const s = await anthropic.messages.stream({ model: 'claude-opus-4-5', max_tokens: 2048, system: SYSTEM, messages });
    for await (const c of s) if (c.type === 'content_block_delta' && c.delta.type === 'text_delta') res.write(`data: ${JSON.stringify({ text: c.delta.text })}\n\n`);
  }
  res.write('data: [DONE]\n\n'); res.end();
}

// ── Cache ─────────────────────────────────────────
const cache = new Map();
function getCache(k) { const h = cache.get(k); return (h && Date.now() - h.ts < h.ttl) ? h.data : null; }
function setCache(k, d, ttl) { cache.set(k, { data: d, ts: Date.now(), ttl }); }
const YFO = { validateResult: false };

// ── Symbols ───────────────────────────────────────
const TICKER_SYMS = [
  { sym: '2330.TW', label: '台積電' }, { sym: '2454.TW', label: '聯發科' },
  { sym: '2317.TW', label: '鴻海'   }, { sym: '3231.TW', label: '緯創'   },
  { sym: '2412.TW', label: '中華電' }, { sym: '^TWII',   label: '台加權' },
  { sym: '^GSPC',   label: 'S&P500' }, { sym: '^IXIC',   label: 'NASDAQ' },
  { sym: '^DJI',    label: '道瓊'   }, { sym: 'GC=F',    label: '黃金'   },
  { sym: 'CL=F',    label: 'WTI油'  }, { sym: 'USDTWD=X',label:'USD/TWD'},
  { sym: '^VIX',    label: 'VIX'    },
];
const WATCH_SYMS = [
  { sym: '2330.TW', code: '2330', name: '台積電', sector: '半導體'  },
  { sym: '2454.TW', code: '2454', name: '聯發科', sector: 'IC設計'  },
  { sym: '2317.TW', code: '2317', name: '鴻海',   sector: '電子製造' },
  { sym: '3231.TW', code: '3231', name: '緯創',   sector: 'AI伺服器' },
  { sym: '2412.TW', code: '2412', name: '中華電', sector: '電信'    },
];

// ── Market Data Routes ────────────────────────────
app.get('/api/ticker', async (_req, res) => {
  const c = getCache('ticker'); if (c) return res.json(c);
  try {
    const quotes = await yf.quote(TICKER_SYMS.map(t => t.sym), {}, YFO);
    const arr = Array.isArray(quotes) ? quotes : [quotes];
    const result = arr.map((q, i) => ({ sym: TICKER_SYMS[i]?.label || q.symbol, ticker: q.symbol, price: q.regularMarketPrice ?? 0, change: q.regularMarketChangePercent ?? 0, raw: q.regularMarketChange ?? 0, up: (q.regularMarketChangePercent ?? 0) >= 0 }));
    setCache('ticker', result, 60000); res.json(result);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/watchlist', async (_req, res) => {
  const c = getCache('watchlist'); if (c) return res.json(c);
  try {
    const quotes = await yf.quote(WATCH_SYMS.map(w => w.sym), {}, YFO);
    const arr = Array.isArray(quotes) ? quotes : [quotes];
    const result = arr.map((q, i) => {
      const m = WATCH_SYMS[i] || {}; const price = q.regularMarketPrice ?? 0; const change = q.regularMarketChangePercent ?? 0;
      return { sym: m.sym, code: m.code, name: m.name, sector: m.sector, price, change, up: change >= 0, open: q.regularMarketOpen ?? price, high: q.regularMarketDayHigh ?? price, low: q.regularMarketDayLow ?? price, volume: q.regularMarketVolume ?? 0, mktCap: q.marketCap ?? null, pe: q.trailingPE ?? null, eps: q.epsTrailingTwelveMonths ?? null, w52Hi: q.fiftyTwoWeekHigh ?? null, w52Lo: q.fiftyTwoWeekLow ?? null };
    });
    setCache('watchlist', result, 60000); res.json(result);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/chart/:symbol', async (req, res) => {
  const { symbol } = req.params; const { period = '6mo', interval = '1d' } = req.query;
  const key = `chart:${symbol}:${period}:${interval}`; const c = getCache(key); if (c) return res.json(c);
  try {
    const days = { '1mo':30,'3mo':90,'6mo':180,'1y':365,'2y':730,'5y':1825 };
    const p1 = new Date(); p1.setDate(p1.getDate() - (days[period] || 180));
    const raw = await yf.historical(symbol, { period1: p1, interval }, YFO);
    const result = raw.map(r => ({ date: r.date instanceof Date ? r.date.toISOString().split('T')[0] : String(r.date), open: r.open, high: r.high, low: r.low, close: r.close, volume: r.volume }));
    setCache(key, result, 300000); res.json(result);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/stock/:symbol', async (req, res) => {
  let sym = req.params.symbol; if (/^\d{4}$/.test(sym)) sym += '.TW';
  const c = getCache('stock:' + sym); if (c) return res.json(c);
  try {
    const [qR, sR] = await Promise.allSettled([yf.quote(sym, {}, YFO), yf.quoteSummary(sym, { modules: ['summaryDetail','defaultKeyStatistics','financialData','assetProfile'] }, YFO)]);
    const q = qR.status === 'fulfilled' ? qR.value : {}; const s = sR.status === 'fulfilled' ? sR.value : {};
    const result = { symbol: sym, name: q.shortName || q.longName || sym, price: q.regularMarketPrice, change: q.regularMarketChange, changePct: q.regularMarketChangePercent, open: q.regularMarketOpen, high: q.regularMarketDayHigh, low: q.regularMarketDayLow, volume: q.regularMarketVolume, mktCap: q.marketCap, pe: q.trailingPE ?? s.summaryDetail?.trailingPE, pb: q.priceToBook ?? s.defaultKeyStatistics?.priceToBook, eps: q.epsTrailingTwelveMonths, divYield: q.dividendYield ?? s.summaryDetail?.dividendYield, roe: s.financialData?.returnOnEquity, description: s.assetProfile?.longBusinessSummary, industry: s.assetProfile?.industry, sector: s.assetProfile?.sector, w52Hi: q.fiftyTwoWeekHigh, w52Lo: q.fiftyTwoWeekLow };
    setCache('stock:' + sym, result, 300000); res.json(result);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/twse/market', async (_req, res) => {
  const c = getCache('twse:market'); if (c) return res.json(c);
  try {
    const r = await fetch('https://www.twse.com.tw/rwd/zh/fund/TWT38U?response=json&selectType=All');
    const result = { fii: null, fetched: new Date().toISOString() };
    if (r.ok) { const d = await r.json(); if (d.data?.length) { const row = d.data[d.data.length - 1]; const buy = parseFloat((row[2]||'0').replace(/,/g,'')); const sell = parseFloat((row[3]||'0').replace(/,/g,'')); result.fii = { date: row[0], buy, sell, net: buy - sell }; } }
    setCache('twse:market', result, 300000); res.json(result);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// ── AI Routes ─────────────────────────────────────
app.post('/api/chat', async (req, res) => {
  try { await streamAI(req.body.messages, req.body.provider || 'claude', res); }
  catch (e) { res.write(`data: ${JSON.stringify({ text: `\n\n⚠️ 錯誤：${e.message}` })}\n\n`); res.write('data: [DONE]\n\n'); res.end(); }
});

app.post('/api/quick-analysis', async (req, res) => {
  const { type, content, provider } = req.body;
  const today = new Date().toLocaleDateString('zh-TW');
  let enriched = content;
  if (type === 'stock' && content) {
    try { const code = content.trim().replace(/[^\d]/g,'').slice(0,4); if (code.length === 4) { const q = await yf.quote(code + '.TW', {}, YFO); enriched = `${content}\n【即時資料 ${today}】股價：${q.regularMarketPrice}元｜漲跌：${q.regularMarketChange?.toFixed(1)}(${q.regularMarketChangePercent?.toFixed(2)}%)｜本益比：${q.trailingPE?.toFixed(1)||'—'}｜市值：${q.marketCap?(q.marketCap/1e8).toFixed(0)+'億':'—'}｜52週高低：${q.fiftyTwoWeekHigh}/${q.fiftyTwoWeekLow}`; } } catch {}
  }
  const prompts = {
    stock:     `請全面分析以下股票（${today}）：\n${enriched}\n\n請提供：\n1.公司簡介\n2.基本面（本益比/殖利率/ROE）\n3.技術面走勢\n4.主要風險\n5.投資評估\n6.建議策略與目標價區間`,
    news:      `請分析以下財經新聞的投資影響（${today}）：\n${content}\n\n1.摘要\n2.受影響產業\n3.短期影響\n4.長期影響\n5.建議策略`,
    portfolio: `請評估以下投資組合（${today}）：\n${content}\n\n1.多元性\n2.風險集中度\n3.預期報酬\n4.潛在風險\n5.優化建議`,
    market:    `請分析台股市場狀況（${today}）：\n1.大盤走勢\n2.強勢族群\n3.弱勢族群\n4.籌碼動向\n5.明日重點\n6.風險提示`,
  };
  try { res.json({ result: await ask(prompts[type] || content, provider) }); }
  catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/valuation', async (req, res) => {
  const { stock, price, pe, pb, roe, eps, growth, sector, provider } = req.body;
  let realData = `用戶提供：股價${price}、PE:${pe}、PB:${pb}、ROE:${roe}%、EPS:${eps}`;
  try { const code = stock.trim().replace(/[^\d]/g,'').slice(0,4); if (code.length === 4) { const q = await yf.quote(code + '.TW', {}, YFO); realData = `【即時資料 ${new Date().toLocaleDateString('zh-TW')}】股價：${q.regularMarketPrice}元｜PE：${q.trailingPE?.toFixed(1)||pe||'—'}｜EPS：${q.epsTrailingTwelveMonths?.toFixed(2)||eps||'—'}｜殖利率：${q.dividendYield?(q.dividendYield*100).toFixed(2)+'%':'—'}｜52週：${q.fiftyTwoWeekHigh}/${q.fiftyTwoWeekLow}`; } } catch {}
  const prompt = `請對「${stock}」進行完整估值分析。\n${realData}\n產業：${sector}，盈餘成長率：${growth}%\n\n請提供：\n1.本益比法合理價\n2.PB法估值\n3.殖利率法\n4.DCF簡化估值\n5.綜合合理價格區間\n6.與同業比較\n7.投資評等與建議買入價\n8.主要風險`;
  try { res.json({ result: await ask(prompt, provider, 3000) }); } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/technical', async (req, res) => {
  const { stock, price, ma5, ma20, ma60, kd_k, kd_d, rsi, volume, trend, provider } = req.body;
  let histCtx = '';
  try { const code = stock.trim().replace(/[^\d]/g,'').slice(0,4); if (code.length === 4) { const p1 = new Date(); p1.setDate(p1.getDate()-20); const hist = await yf.historical(code + '.TW', { period1: p1, interval: '1d' }, YFO); const last5 = hist.slice(-5).map(h => { const dt = h.date instanceof Date ? h.date.toISOString().split('T')[0] : String(h.date); return `${dt} 收${h.close?.toFixed(0)} 量${h.volume?(h.volume/1e6).toFixed(1)+'M':'—'}`; }).join('\n'); if (last5) histCtx = `\n【近5日歷史】\n${last5}`; } } catch {}
  const prompt = `請對「${stock}」進行技術指標分析（${new Date().toLocaleDateString('zh-TW')}）。\n股價：${price}｜MA5/20/60：${ma5}/${ma20}/${ma60}\nKD K/D：${kd_k}/${kd_d}｜RSI：${rsi}｜成交量：${volume}｜走勢：${trend}${histCtx}\n\n請分析：\n1.均線多空排列\n2.KD指標解讀\n3.RSI強弱\n4.支撐位與壓力位\n5.短中長期趨勢\n6.操作建議（進場點/停損點/目標價）`;
  try { res.json({ result: await ask(prompt, provider, 2500) }); } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/backtest', async (req, res) => {
  const { strategy, stock, period, capital, buyRule, sellRule, provider } = req.body;
  const prompt = `請對以下投資策略進行回測模擬（${new Date().toLocaleDateString('zh-TW')}）：\n標的：${stock}｜策略：${strategy}｜期間：${period}｜資金：${capital}\n買入：${buyRule}\n賣出：${sellRule}\n\n請提供：\n1.策略邏輯評估\n2.模擬回測結果（年化報酬/最大回撤/勝率）\n3.優點\n4.缺點與失效情境\n5.改進建議\n6.與大盤比較\n7.適合市場環境\n注意：此為AI模擬分析。`;
  try { res.json({ result: await ask(prompt, provider, 3000) }); } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/allocation', async (req, res) => {
  const { holdings, risk, goal, horizon, provider } = req.body;
  const prompt = `請分析並優化投資組合（${new Date().toLocaleDateString('zh-TW')}）：\n持股：\n${holdings}\n風險：${risk}｜目標：${goal}｜期間：${horizon}\n\n請提供：\n1.組合健診\n2.產業/地區配置\n3.建議最佳化比例\n4.減碼/增碼說明\n5.建議加入標的\n6.預期改善效益`;
  try { res.json({ result: await ask(prompt, provider, 3000) }); } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/report', async (req, res) => {
  const { portfolio, period, focus, provider } = req.body;
  const prompt = `請為以下投資組合生成「${period}投資分析報告」（${new Date().toLocaleDateString('zh-TW')}）：\n持股：\n${portfolio}\n重點：${focus}\n\n報告章節：\n一、執行摘要\n二、市場環境回顧\n三、持股績效分析\n四、風險指標評估\n五、重要事件影響\n六、下期展望與策略\n七、調整建議\n八、免責聲明`;
  try { res.json({ result: await ask(prompt, provider, 4096) }); } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/monthly', async (req, res) => {
  const { month, provider } = req.body;
  const prompt = `請生成「${month}台股市場重點摘要推播」：\n# ${month}投資摘要\n## 大盤表現\n## 產業亮點（前3強勢/弱勢）\n## 重要財經事件\n## 籌碼動向（外資/投信/自營）\n## 本月最強個股\n## 下月關鍵觀察指標\n## AI投資建議重點\n## 風險警示`;
  try { res.json({ result: await ask(prompt, provider, 3000) }); } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));
app.listen(PORT, () => console.log(`✅ 智投 AI 已啟動 Port: ${PORT}`));
