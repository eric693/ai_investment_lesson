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

/* ── AI Clients ─────────────────────────────────── */
const anthropic = process.env.ANTHROPIC_API_KEY
  ? new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY }) : null;
const openai = process.env.OPENAI_API_KEY
  ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;
const FRED_KEY = process.env.FRED_API_KEY || null;

const SYSTEM = `你是一位專業的 AI 股票理財顧問「智投 AI」，專精台股與全球市場。
請用繁體中文回覆，條理清晰，善用表格與條列式。數據具體，語氣專業但親切。
所有分析僅供參考，不構成投資建議，投資人應自行評估風險。`;

const YFO = { validateResult: false };

/* ── Cache ───────────────────────────────────────── */
const _cache = new Map();
function getCache(key) {
  const h = _cache.get(key);
  if (h && Date.now() - h.ts < h.ttl) return h.data;
  return null;
}
function setCache(key, data, ttl) { _cache.set(key, { data, ts: Date.now(), ttl }); }

/* ── Model availability ──────────────────────────── */
app.get('/api/models', (_req, res) => res.json({
  claude: !!anthropic, openai: !!openai, fred: !!FRED_KEY,
  default: anthropic ? 'claude' : (openai ? 'openai' : null),
}));

/* ── AI helpers ──────────────────────────────────── */
async function ask(prompt, provider = 'claude', maxTokens = 2048) {
  if (provider === 'openai') {
    if (!openai) throw new Error('OpenAI API key 未設定');
    const r = await openai.chat.completions.create({
      model: 'gpt-4o', max_tokens: maxTokens,
      messages: [{ role: 'system', content: SYSTEM }, { role: 'user', content: prompt }],
    });
    return r.choices[0].message.content;
  }
  if (!anthropic) throw new Error('Anthropic API key 未設定');
  const r = await anthropic.messages.create({
    model: 'claude-opus-4-5', max_tokens: maxTokens, system: SYSTEM,
    messages: [{ role: 'user', content: prompt }],
  });
  return r.content[0].text;
}

async function streamAI(messages, provider = 'claude', res) {
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
      const t = chunk.choices[0]?.delta?.content || '';
      if (t) res.write(`data: ${JSON.stringify({ text: t })}\n\n`);
    }
  } else {
    if (!anthropic) throw new Error('Anthropic API key 未設定');
    const s = await anthropic.messages.stream({ model: 'claude-opus-4-5', max_tokens: 2048, system: SYSTEM, messages });
    for await (const c of s)
      if (c.type === 'content_block_delta' && c.delta.type === 'text_delta')
        res.write(`data: ${JSON.stringify({ text: c.delta.text })}\n\n`);
  }
  res.write('data: [DONE]\n\n');
  res.end();
}

/* ════════════════════════════════════════════════════
   REAL MARKET DATA
════════════════════════════════════════════════════ */

const TICKER_LIST = [
  { sym: '2330.TW', label: '台積電' }, { sym: '2454.TW', label: '聯發科' },
  { sym: '2317.TW', label: '鴻海'   }, { sym: '^TWII',   label: '台加權' },
  { sym: '^GSPC',   label: 'S&P500' }, { sym: '^IXIC',   label: 'NASDAQ' },
  { sym: '^DJI',    label: '道瓊'   }, { sym: '^VIX',    label: 'VIX'    },
  { sym: 'GC=F',    label: '黃金'   }, { sym: 'CL=F',    label: 'WTI油'  },
  { sym: 'USDTWD=X',label: 'USD/TWD'},
];

app.get('/api/ticker', async (_req, res) => {
  const cached = getCache('ticker');
  if (cached) return res.json(cached);
  try {
    const quotes = await yf.quote(TICKER_LIST.map(t => t.sym), {}, YFO);
    const arr = Array.isArray(quotes) ? quotes : [quotes];
    const result = arr.map((q, i) => ({
      sym: TICKER_LIST[i]?.label || q.symbol, ticker: q.symbol,
      price: q.regularMarketPrice ?? 0,
      change: q.regularMarketChangePercent ?? 0,
      up: (q.regularMarketChangePercent ?? 0) >= 0,
    }));
    setCache('ticker', result, 60_000);
    res.json(result);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

const WATCHLIST_LIST = [
  { sym: '2330.TW', code: '2330', name: '台積電', sector: '半導體'   },
  { sym: '2454.TW', code: '2454', name: '聯發科', sector: 'IC設計'   },
  { sym: '2317.TW', code: '2317', name: '鴻海',   sector: '電子製造'  },
  { sym: '3231.TW', code: '3231', name: '緯創',   sector: 'AI 伺服器' },
  { sym: '2412.TW', code: '2412', name: '中華電', sector: '電信'     },
];

app.get('/api/watchlist', async (_req, res) => {
  const cached = getCache('watchlist');
  if (cached) return res.json(cached);
  try {
    const quotes = await yf.quote(WATCHLIST_LIST.map(w => w.sym), {}, YFO);
    const arr = Array.isArray(quotes) ? quotes : [quotes];
    const result = arr.map((q, i) => {
      const m = WATCHLIST_LIST[i] || {};
      return {
        sym: m.sym, code: m.code, name: m.name, sector: m.sector,
        price: q.regularMarketPrice ?? 0,
        change: q.regularMarketChangePercent ?? 0,
        up: (q.regularMarketChangePercent ?? 0) >= 0,
        volume: q.regularMarketVolume ?? 0,
        pe: q.trailingPE ?? null, mktCap: q.marketCap ?? null,
        week52H: q.fiftyTwoWeekHigh ?? null, week52L: q.fiftyTwoWeekLow ?? null,
      };
    });
    setCache('watchlist', result, 60_000);
    res.json(result);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/chart/:symbol', async (req, res) => {
  const { symbol } = req.params;
  const period = req.query.period || '6mo';
  const interval = req.query.interval || '1d';
  const key = `chart:${symbol}:${period}:${interval}`;
  const cached = getCache(key);
  if (cached) return res.json(cached);
  try {
    const days = { '1mo':30,'3mo':90,'6mo':180,'1y':365,'2y':730,'5y':1825 };
    const p1 = new Date(); p1.setDate(p1.getDate() - (days[period] || 180));
    const raw = await yf.historical(symbol, { period1: p1, interval }, YFO);
    const result = raw.map(d => ({
      date: d.date.toISOString().split('T')[0],
      open: d.open, high: d.high, low: d.low, close: d.close, volume: d.volume,
    }));
    setCache(key, result, 300_000);
    res.json(result);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/twse/market', async (_req, res) => {
  const cached = getCache('twse:market');
  if (cached) return res.json(cached);
  try {
    const fiiRes = await fetch('https://www.twse.com.tw/rwd/zh/fund/TWT38U?response=json&selectType=All');
    const result = { fii: null, fetched: new Date().toISOString() };
    if (fiiRes.ok) {
      const d = await fiiRes.json();
      if (d.data?.length) {
        const row  = d.data[d.data.length - 1];
        const buy  = parseFloat((row[2]||'0').replace(/,/g,''));
        const sell = parseFloat((row[3]||'0').replace(/,/g,''));
        result.fii = { date: row[0], buy, sell, net: buy - sell };
      }
    }
    setCache('twse:market', result, 300_000);
    res.json(result);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/stock/:symbol', async (req, res) => {
  const raw = req.params.symbol;
  const sym = /^\d{4}$/.test(raw) ? raw + '.TW' : raw;
  const cached = getCache('stock:' + sym);
  if (cached) return res.json(cached);
  try {
    const [qR, sR] = await Promise.allSettled([
      yf.quote(sym, {}, YFO),
      yf.quoteSummary(sym, { modules: ['summaryDetail','defaultKeyStatistics','financialData','assetProfile'] }, YFO),
    ]);
    const q = qR.status === 'fulfilled' ? qR.value : {};
    const s = sR.status === 'fulfilled' ? sR.value : {};
    const result = {
      symbol: sym, name: q.shortName || q.longName || sym,
      price: q.regularMarketPrice, change: q.regularMarketChange,
      changePct: q.regularMarketChangePercent, volume: q.regularMarketVolume,
      mktCap: q.marketCap, pe: q.trailingPE ?? s.summaryDetail?.trailingPE,
      pb: q.priceToBook ?? s.defaultKeyStatistics?.priceToBook,
      eps: q.epsTrailingTwelveMonths, divYield: q.dividendYield ?? s.summaryDetail?.dividendYield,
      roe: s.financialData?.returnOnEquity, week52H: q.fiftyTwoWeekHigh, week52L: q.fiftyTwoWeekLow,
      industry: s.assetProfile?.industry, sector: s.assetProfile?.sector, desc: s.assetProfile?.longBusinessSummary,
    };
    setCache('stock:' + sym, result, 120_000);
    res.json(result);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

/* ════════════════════════════════════════════════════
   NEW MODULE 1: MACRO ECONOMIC DATA
════════════════════════════════════════════════════ */

// Helper: fetch single FRED series (requires free API key from fred.stlouisfed.org)
async function fetchFRED(seriesId, limit = 3) {
  if (!FRED_KEY) return null;
  try {
    const url = `https://api.stlouisfed.org/fred/series/observations?series_id=${seriesId}&api_key=${FRED_KEY}&file_type=json&limit=${limit}&sort_order=desc`;
    const r = await fetch(url);
    const d = await r.json();
    return d.observations?.filter(o => o.value !== '.').slice(0, limit) || null;
  } catch { return null; }
}

// Helper: BLS NFP (no key needed)
async function fetchNFP() {
  try {
    const year = new Date().getFullYear();
    const r = await fetch('https://api.bls.gov/publicAPI/v2/timeseries/data/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ seriesid: ['CES0000000001'], startyear: String(year - 1), endyear: String(year) }),
    });
    const d = await r.json();
    const series = d.Results?.series?.[0]?.data || [];
    return series.slice(0, 3).map(s => ({
      date: `${s.year}-${s.period.replace('M','')}`,
      value: s.value,
    }));
  } catch { return null; }
}

app.get('/api/macro', async (_req, res) => {
  const cached = getCache('macro');
  if (cached) return res.json(cached);

  try {
    // Parallel fetch all macro indicators
    const [fedRate, cpi, pce, unemployment, vixQ, treasury10y, nfp] = await Promise.allSettled([
      fetchFRED('FEDFUNDS', 3),       // Fed Funds Rate
      fetchFRED('CPIAUCSL', 3),       // CPI All Items
      fetchFRED('PCEPI', 3),          // PCE Price Index (Fed's preferred)
      fetchFRED('UNRATE', 3),         // Unemployment Rate
      yf.quote('^VIX', {}, YFO),     // VIX current
      fetchFRED('DGS10', 3),          // 10-Year Treasury Yield
      fetchNFP(),                      // Non-Farm Payrolls
    ]);

    const result = {
      fedRate:      fedRate.status === 'fulfilled' ? fedRate.value : null,
      cpi:          cpi.status === 'fulfilled' ? cpi.value : null,
      pce:          pce.status === 'fulfilled' ? pce.value : null,
      unemployment: unemployment.status === 'fulfilled' ? unemployment.value : null,
      vix:          vixQ.status === 'fulfilled' ? {
        value: vixQ.value.regularMarketPrice,
        change: vixQ.value.regularMarketChangePercent,
      } : null,
      treasury10y:  treasury10y.status === 'fulfilled' ? treasury10y.value : null,
      nfp:          nfp.status === 'fulfilled' ? nfp.value : null,
      fredAvailable: !!FRED_KEY,
      fetched: new Date().toISOString(),
    };

    setCache('macro', result, 3_600_000); // 1 hour TTL (macro data changes slowly)
    res.json(result);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/macro/analyze', async (req, res) => {
  const { macroData, provider } = req.body;

  const fedRateStr = macroData.fedRate?.[0]
    ? `聯邦基金利率：${macroData.fedRate[0].value}%（前次：${macroData.fedRate[1]?.value || '—'}%）`
    : '聯邦基金利率：資料未提供（需設定 FRED_API_KEY）';

  const cpiStr = macroData.cpi?.[0]
    ? `CPI（消費者物價指數）：${macroData.cpi[0].value}（前次：${macroData.cpi[1]?.value || '—'}）`
    : 'CPI：資料未提供';

  const unemployStr = macroData.unemployment?.[0]
    ? `失業率：${macroData.unemployment[0].value}%`
    : '失業率：資料未提供';

  const vixStr = macroData.vix
    ? `VIX 恐慌指數：${macroData.vix.value?.toFixed(2)}（${macroData.vix.change?.toFixed(2)}%）`
    : 'VIX：資料未提供';

  const nfpStr = macroData.nfp?.[0]
    ? `非農就業人數：${macroData.nfp[0].value}千人（${macroData.nfp[0].date}）`
    : '非農就業：資料未提供';

  const treasury10yStr = macroData.treasury10y?.[0]
    ? `10年期美債殖利率：${macroData.treasury10y[0].value}%`
    : '10年期美債：資料未提供';

  const prompt = `請根據以下總體經濟指標，進行完整的宏觀經濟環境分析，判斷目前經濟週期位置：

【即時總經數據】
${fedRateStr}
${cpiStr}
${unemployStr}
${vixStr}
${nfpStr}
${treasury10yStr}

請提供以下分析：

一、經濟週期判斷
判斷目前處於「擴張期」、「過熱期」、「衰退期」還是「復甦期」，並說明判斷依據。

二、聯準會政策展望
根據 Fed 點陣圖邏輯與通膨、就業數據，判斷未來 6-12 個月利率走向（升息/暫停/降息），並說明對股市的影響。

三、通膨與就業分析
CPI 與 PCE 的趨勢是否收斂？非農就業是否顯示勞動市場過熱或降溫？

四、市場風險評估
VIX 目前水準代表市場恐慌程度如何？10 年期美債殖利率對股市估值有何壓力？

五、台股投資含義
目前總經環境對台股（尤其是科技類股、出口導向企業）的影響為何？

六、建議配置方向
在此總經背景下，建議增持哪些資產類別？減持哪些？（股票/債券/黃金/現金比例建議）

七、需要關注的風險事件
未來 3 個月內有哪些重大總經事件或數據發布需要關注？`;

  try {
    res.json({ result: await ask(prompt, provider, 3000) });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

/* ════════════════════════════════════════════════════
   NEW MODULE 2: BLACK SWAN WARNING SYSTEM
   Rolling Z-Score anomaly detection on VIX + volume
════════════════════════════════════════════════════ */

// Calculate rolling statistics
function rollingStats(data, window) {
  const stats = [];
  for (let i = 0; i < data.length; i++) {
    if (i < window - 1) { stats.push({ mean: null, std: null }); continue; }
    const slice = data.slice(i - window + 1, i + 1);
    const mean = slice.reduce((a, b) => a + b, 0) / window;
    const variance = slice.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / window;
    stats.push({ mean, std: Math.sqrt(variance) });
  }
  return stats;
}

// Z-score anomaly detection
function detectAnomalies(values, dates, threshold = 2.5, window = 20) {
  const stats = rollingStats(values, window);
  const anomalies = [];

  for (let i = window; i < values.length; i++) {
    const { mean, std } = stats[i];
    if (!mean || !std || std === 0) continue;
    const zscore = (values[i] - mean) / std;
    if (Math.abs(zscore) >= threshold) {
      anomalies.push({
        date: dates[i],
        value: values[i],
        zscore: parseFloat(zscore.toFixed(2)),
        mean: parseFloat(mean.toFixed(2)),
        direction: zscore > 0 ? 'spike' : 'drop',
      });
    }
  }
  return anomalies;
}

app.post('/api/blackswan', async (req, res) => {
  const { symbols = ['2330.TW', '2454.TW'], provider } = req.body;
  const cached = getCache('blackswan:' + symbols.join(','));
  if (cached) return res.json(cached);

  try {
    const p1 = new Date(); p1.setDate(p1.getDate() - 252); // ~1 year

    // Fetch VIX and target stocks in parallel
    const fetchTargets = ['^VIX', ...symbols.slice(0, 3)].map(sym =>
      yf.historical(sym, { period1: p1, interval: '1d' }, YFO)
        .then(data => ({ sym, data }))
        .catch(() => ({ sym, data: [] }))
    );

    const results = await Promise.all(fetchTargets);

    const analysis = {};

    for (const { sym, data } of results) {
      if (!data || data.length < 30) continue;

      const dates  = data.map(d => d.date.toISOString().split('T')[0]);
      const closes = data.map(d => d.close || 0);
      const volumes = data.map(d => d.volume || 0);

      // Price anomalies (daily return z-score)
      const returns = closes.slice(1).map((c, i) => ((c - closes[i]) / closes[i]) * 100);
      const returnDates = dates.slice(1);
      const priceAnomalies = detectAnomalies(returns, returnDates, 2.5, 20);

      // Volume anomalies
      const volAnomalies = detectAnomalies(volumes, dates, 2.5, 20);

      // Current z-score
      const recentReturns = returns.slice(-21);
      const latestReturn  = recentReturns[recentReturns.length - 1];
      const mean = recentReturns.slice(0, -1).reduce((a, b) => a + b, 0) / 20;
      const std  = Math.sqrt(recentReturns.slice(0, -1).reduce((a, b) => a + Math.pow(b - mean, 2), 0) / 20);
      const currentZscore = std > 0 ? parseFloat(((latestReturn - mean) / std).toFixed(2)) : 0;

      // Recent vol z-score
      const recentVols = volumes.slice(-21);
      const latestVol  = recentVols[recentVols.length - 1];
      const volMean = recentVols.slice(0, -1).reduce((a, b) => a + b, 0) / 20;
      const volStd  = Math.sqrt(recentVols.slice(0, -1).reduce((a, b) => a + Math.pow(b - volMean, 2), 0) / 20);
      const volZscore = volStd > 0 ? parseFloat(((latestVol - volMean) / volStd).toFixed(2)) : 0;

      analysis[sym] = {
        currentZscore,
        volZscore,
        recentAnomalies: priceAnomalies.slice(-5),   // last 5 price anomalies
        recentVolAnomalies: volAnomalies.slice(-5),   // last 5 volume anomalies
        totalAnomalies: priceAnomalies.length,
        riskLevel: Math.abs(currentZscore) > 3 ? 'HIGH' :
                   Math.abs(currentZscore) > 2 ? 'MEDIUM' : 'LOW',
        latestClose: closes[closes.length - 1],
        latestDate: dates[dates.length - 1],
      };
    }

    // Build AI prompt with the anomaly data
    const summaryLines = Object.entries(analysis).map(([sym, a]) => {
      const label = sym === '^VIX' ? 'VIX 恐慌指數' : sym;
      return `${label}：目前Z值=${a.currentZscore}，成交量Z值=${a.volZscore}，風險等級=${a.riskLevel}，近期異常次數=${a.totalAnomalies}次（過去252日）`;
    }).join('\n');

    const aiPrompt = `請根據以下市場異常偵測結果（基於滾動Z值分析），進行黑天鵝風險評估：

【異常偵測結果（Z值臨界值 ±2.5）】
${summaryLines}

Z值說明：
- Z值 = (當前值 - 20日滾動均值) / 20日滾動標準差
- |Z| > 2.5 視為統計異常
- |Z| > 3.0 為極端異常（3σ事件）

請提供：

一、整體風險評級
綜合所有指標，目前市場風險等級為何（低/中/高/極高）？判斷依據？

二、VIX 分析
目前恐慌指數的 Z 值意味著什麼？是否有黑天鵝前兆？

三、個股異常解讀
哪些股票出現了顯著的價格或成交量異常？可能的原因？

四、歷史類比
Z 值達到此水準時，歷史上常見的後續市場發展為何？

五、具體預警建議
- 是否建議降低倉位？
- 是否建議增加避險部位（如黃金、債券、反向 ETF）？
- 停損點建議
- 需要持續監控的關鍵指標

六、風險緩解策略
在高風險環境下，如何調整投資組合以因應潛在的黑天鵝事件？`;

    const aiAnalysis = await ask(aiPrompt, provider, 3000);

    const finalResult = { analysis, aiAnalysis, fetched: new Date().toISOString() };
    setCache('blackswan:' + symbols.join(','), finalResult, 300_000);
    res.json(finalResult);
  } catch (e) {
    console.error('blackswan error:', e.message);
    res.status(500).json({ error: e.message });
  }
});

/* ════════════════════════════════════════════════════
   MODULE 3: ENHANCED VALUATION (DCF + PE Band)
════════════════════════════════════════════════════ */

app.post('/api/valuation', async (req, res) => {
  const { stock, price, pe, pb, roe, eps, growth, sector, provider } = req.body;

  let realData = `用戶提供：股價${price}、PE${pe}、PB${pb}、ROE${roe}%、EPS${eps}、成長率${growth}%`;
  let dcfData = '';
  let peBandData = '';

  try {
    const code = stock.match(/\d{4}/)?.[0];
    if (code) {
      const sym = code + '.TW';

      // Get current quote + fundamentals
      const [qR, sR] = await Promise.allSettled([
        yf.quote(sym, {}, YFO),
        yf.quoteSummary(sym, { modules: ['financialData','summaryDetail','defaultKeyStatistics','earnings'] }, YFO),
      ]);
      const q = qR.status === 'fulfilled' ? qR.value : {};
      const s = sR.status === 'fulfilled' ? sR.value : {};

      realData = `【Yahoo Finance 即時資料】
股價：${q.regularMarketPrice} | PE：${q.trailingPE?.toFixed(1)||pe||'—'} | EPS：${q.epsTrailingTwelveMonths?.toFixed(2)||eps||'—'}
殖利率：${q.dividendYield?(q.dividendYield*100).toFixed(2)+'%':'—'} | ROE：${s.financialData?.returnOnEquity?(s.financialData.returnOnEquity*100).toFixed(1)+'%':roe+'%'}
市值：${q.marketCap?(q.marketCap/1e8).toFixed(0)+'億':'—'} | 52週高低：${q.fiftyTwoWeekHigh}/${q.fiftyTwoWeekLow}
自由現金流：${s.financialData?.freeCashflow?(s.financialData.freeCashflow/1e8).toFixed(1)+'億':'—'}
毛利率：${s.financialData?.grossMargins?(s.financialData.grossMargins*100).toFixed(1)+'%':'—'}`;

      // Fetch 5-year price + EPS history for PE Band
      const p1 = new Date(); p1.setFullYear(p1.getFullYear() - 5);
      const [histR] = await Promise.allSettled([
        yf.historical(sym, { period1: p1, interval: '1mo' }, YFO),
      ]);

      if (histR.status === 'fulfilled' && histR.value.length > 0) {
        const hist = histR.value;
        const prices = hist.map(h => h.close).filter(Boolean);
        const priceMin = Math.min(...prices);
        const priceMax = Math.max(...prices);
        const priceMedian = prices.sort((a,b)=>a-b)[Math.floor(prices.length/2)];
        const currentPrice = q.regularMarketPrice || parseFloat(price);
        const currentEPS = q.epsTrailingTwelveMonths || parseFloat(eps);

        peBandData = `\n【PE Band 分析（近5年）】
歷史價格區間：${priceMin?.toFixed(0)} ~ ${priceMax?.toFixed(0)} 元
歷史價格中位數：${priceMedian?.toFixed(0)} 元
目前股價：${currentPrice} 元
目前股價位於5年歷史的：${((currentPrice - priceMin)/(priceMax - priceMin)*100).toFixed(1)}% 分位
目前本益比：${q.trailingPE?.toFixed(1) || pe}
5年最高本益比（估）：${(priceMax / (currentEPS||1)).toFixed(1)}
5年最低本益比（估）：${(priceMin / (currentEPS||1)).toFixed(1)}`;

        // DCF data
        const fcf = s.financialData?.freeCashflow;
        const growthRate = parseFloat(growth) / 100 || 0.1;
        if (fcf) {
          const terminalGrowth = 0.025;
          const wacc = 0.10;
          let dcfValue = 0;
          let projectedFCF = fcf;
          const projections = [];
          for (let yr = 1; yr <= 5; yr++) {
            const g = growthRate * (1 - (yr-1) * 0.05);  // gradually declining growth
            projectedFCF = projectedFCF * (1 + Math.max(g, terminalGrowth));
            const pv = projectedFCF / Math.pow(1 + wacc, yr);
            dcfValue += pv;
            projections.push(`第${yr}年：FCF 預估 ${(projectedFCF/1e8).toFixed(1)}億，現值 ${(pv/1e8).toFixed(1)}億`);
          }
          const terminalValue = projectedFCF * (1 + terminalGrowth) / (wacc - terminalGrowth);
          const terminalPV = terminalValue / Math.pow(1 + wacc, 5);
          dcfValue += terminalPV;
          const sharesOutstanding = q.marketCap ? q.marketCap / currentPrice : null;
          const dcfPerShare = sharesOutstanding ? dcfValue / sharesOutstanding : null;

          dcfData = `\n【DCF 現金流折現模型（WACC=${(wacc*100).toFixed(0)}%，終值成長率=2.5%）】
近期自由現金流：${(fcf/1e8).toFixed(1)} 億元
預估盈餘成長率（前5年）：${growth}%（逐年遞減）
${projections.join('\n')}
終值現值：${(terminalPV/1e8).toFixed(1)} 億
DCF 總估算價值：${(dcfValue/1e8).toFixed(1)} 億
${dcfPerShare ? `每股 DCF 價值（估）：${dcfPerShare.toFixed(0)} 元` : '（需股數資料計算每股價值）'}`;
        }
      }
    }
  } catch (e) { console.warn('valuation data fetch:', e.message); }

  const prompt = `請對「${stock}」進行完整多方法估值分析。

${realData}
${peBandData}
${dcfData}

請提供：

一、多種估值法比較
1. 本益比法（PE）：歷史區間合理價格計算
2. 股價淨值比法（PB）：合理 PB 區間與對應價格
3. 殖利率法：殖利率回歸至合理水準的對應股價
4. DCF 現金流折現：根據上述 DCF 數據解讀每股內在價值
5. PEG 成長調整法：本益比除以成長率的合理性評估

二、PE Band 位階判斷
目前本益比在歷史區間的相對位置意義：是偏貴還是偏便宜？與歷史均值的距離？

三、安全邊際評估
綜合各方法的合理價格區間，目前股價的安全邊際為何？（有多少下行保護空間）

四、買入區間建議
建議的「強力買入價」、「合理買入價」、「觀察等待價」各為何？

五、主要風險因子
可能造成估值下修的 3 個主要風險。

六、投資建議
買入 / 持有 / 賣出，並說明理由。

所屬產業：${sector}，用戶提供成長率：${growth}%，PB：${pb}`;

  try { res.json({ result: await ask(prompt, provider, 3500) }); }
  catch (e) { res.status(500).json({ error: e.message }); }
});

/* ════════════════════════════════════════════════════
   MODULE 4: ENHANCED BACKTEST (Sharpe + Max Drawdown)
════════════════════════════════════════════════════ */

// Calculate Sharpe Ratio
function calcSharpe(returns, riskFreeRate = 0.02) {
  if (returns.length === 0) return null;
  const annualizedRF = riskFreeRate / 252;
  const excessReturns = returns.map(r => r - annualizedRF);
  const mean = excessReturns.reduce((a, b) => a + b, 0) / excessReturns.length;
  const variance = excessReturns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / excessReturns.length;
  const std = Math.sqrt(variance);
  if (std === 0) return null;
  return parseFloat(((mean / std) * Math.sqrt(252)).toFixed(3));
}

// Calculate Max Drawdown
function calcMaxDrawdown(prices) {
  if (prices.length === 0) return null;
  let peak = prices[0];
  let maxDD = 0;
  let peakDate = 0;
  let troughDate = 0;
  let ddPeakDate = 0;
  let ddTroughDate = 0;

  for (let i = 1; i < prices.length; i++) {
    if (prices[i] > peak) {
      peak = prices[i];
      peakDate = i;
    }
    const dd = (prices[i] - peak) / peak;
    if (dd < maxDD) {
      maxDD = dd;
      ddPeakDate = peakDate;
      ddTroughDate = i;
    }
  }
  return parseFloat((maxDD * 100).toFixed(2)); // as percentage
}

// Calculate Calmar Ratio (annualized return / max drawdown)
function calcCalmar(annualReturn, maxDrawdown) {
  if (!maxDrawdown || maxDrawdown === 0) return null;
  return parseFloat((annualReturn / Math.abs(maxDrawdown)).toFixed(3));
}

app.post('/api/backtest', async (req, res) => {
  const { strategy, stock, period, capital, buyRule, sellRule, provider } = req.body;

  // Fetch real historical data for quantitative metrics
  let metricsData = '';
  let realMetrics = null;

  try {
    const code = stock.match(/\d{4}/)?.[0];
    const sym = code ? code + '.TW' : stock === '0050' ? '0050.TW' : '^TWII';
    const periodDays = { '近 1 年':365, '近 3 年':1095, '近 5 年':1825, '近 10 年':3650, '2020-2024':1825 };
    const days = periodDays[period] || 1095;
    const p1 = new Date(); p1.setDate(p1.getDate() - days);

    const hist = await yf.historical(sym, { period1: p1, interval: '1d' }, YFO);

    if (hist && hist.length > 30) {
      const prices  = hist.map(d => d.close).filter(Boolean);
      const dates   = hist.map(d => d.date.toISOString().split('T')[0]);

      // Daily returns
      const returns = prices.slice(1).map((p, i) => (p - prices[i]) / prices[i]);

      // Annualized return
      const totalReturn = (prices[prices.length-1] - prices[0]) / prices[0];
      const years = days / 365;
      const annualReturn = (Math.pow(1 + totalReturn, 1/years) - 1) * 100;

      // Metrics
      const sharpe   = calcSharpe(returns);
      const maxDD    = calcMaxDrawdown(prices);
      const calmar   = calcCalmar(annualReturn, maxDD);
      const winDays  = returns.filter(r => r > 0).length;
      const winRate  = ((winDays / returns.length) * 100).toFixed(1);

      // Volatility (annualized)
      const meanR = returns.reduce((a,b) => a+b, 0) / returns.length;
      const variance = returns.reduce((a,b) => a + Math.pow(b-meanR, 2), 0) / returns.length;
      const annualVol = (Math.sqrt(variance) * Math.sqrt(252) * 100).toFixed(2);

      // Drawdown periods analysis
      let inDrawdown = false;
      let peak = prices[0];
      let ddCount = 0;
      for (const p of prices) {
        if (p > peak) { peak = p; inDrawdown = false; }
        else if ((p - peak)/peak < -0.05 && !inDrawdown) { ddCount++; inDrawdown = true; }
      }

      realMetrics = { sharpe, maxDD, calmar, winRate, annualReturn: annualReturn.toFixed(2), annualVol, ddCount, totalReturn: (totalReturn*100).toFixed(2), dataPoints: prices.length };

      metricsData = `
【基礎市場指標（${sym}，${period}，${hist.length} 個交易日實際數據）】
期間總報酬：${(totalReturn*100).toFixed(2)}%
年化報酬率：${annualReturn.toFixed(2)}%
夏普比率（Sharpe Ratio）：${sharpe ?? '資料不足'}（風險調整後報酬，>1 為優秀，>2 為卓越）
最大回撤（Max Drawdown）：${maxDD}%（持有期間最大虧損幅度）
卡瑪比率（Calmar Ratio）：${calmar ?? '—'}（年化報酬 / 最大回撤，>1 為佳）
年化波動率：${annualVol}%
每日上漲勝率：${winRate}%
重大回撤次數（>5%）：${ddCount} 次`;
    }
  } catch (e) { console.warn('backtest data fetch:', e.message); }

  const prompt = `請對以下投資策略進行深度回測分析模擬。

【策略設定】
標的：${stock} | 策略名稱：${strategy}
回測期間：${period} | 初始資金：${capital}
買入條件：${buyRule}
賣出條件：${sellRule}
${metricsData}

請提供：

一、量化績效模擬
根據策略邏輯，模擬以下指標（參考上方真實市場數據進行合理調整）：
- 預估策略年化報酬率（vs 被動持有的 ${realMetrics?.annualReturn || '—'}%）
- 夏普比率（Sharpe Ratio）：策略風險調整後報酬估計
  * 夏普值 < 0：策略不如無風險利率
  * 夏普值 0-1：報酬不佳
  * 夏普值 1-2：良好
  * 夏普值 > 2：優秀
- 最大回撤（Max Drawdown）：策略最壞情況下的最大虧損
- 勝率（%）：獲利交易次數佔總交易次數的比例
- 盈虧比：平均獲利 / 平均虧損

二、回測失敗案例分析（最重要）
這個策略在什麼市場環境下會徹底失效？
請列出 3 種具體的失敗情境：
情境1：（例：橫盤整理市場的假突破）
情境2：（例：黑天鵝事件造成跳空缺口跌破停損）
情境3：（例：趨勢轉折前的過度交易）

對每種失敗情境，說明：
- 失敗的根本原因
- 歷史上發生此情境的時間點（如有）
- 如何改進策略以因應

三、與大盤比較
vs 被動持有該標的：
- 多頭市場表現比較
- 空頭市場表現比較
- 橫盤市場表現比較

四、策略優化建議
列出 3 個具體可以提升夏普比率的改進方向：
1. 
2. 
3. 

五、適合與不適合的市場環境
適合：
不適合：

六、風險管理建議
對此策略，建議的部位大小（佔總資金比例）、最大單筆虧損容忍度？

注意：量化指標為 AI 模擬估算，非實際程式化回測結果。`;

  try {
    const result = await ask(prompt, provider, 3500);
    res.json({ result, metrics: realMetrics });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

/* ════════════════════════════════════════════════════
   EXISTING AI ENDPOINTS (unchanged)
════════════════════════════════════════════════════ */

app.post('/api/chat', async (req, res) => {
  try { await streamAI(req.body.messages, req.body.provider || 'claude', res); }
  catch (e) {
    res.write(`data: ${JSON.stringify({ text: `\n\n⚠️ 錯誤：${e.message}` })}\n\n`);
    res.write('data: [DONE]\n\n'); res.end();
  }
});

app.post('/api/quick-analysis', async (req, res) => {
  const { type, content, provider } = req.body;
  let enriched = content;
  if (type === 'stock' && content) {
    try {
      const code = content.trim().match(/\d{4}/)?.[0];
      if (code) {
        const q = await yf.quote(code + '.TW', {}, YFO);
        enriched = `${content}\n【即時資料】股價：${q.regularMarketPrice} 漲跌：${q.regularMarketChangePercent?.toFixed(2)}% PE：${q.trailingPE?.toFixed(1)||'—'} 52週高低：${q.fiftyTwoWeekHigh}/${q.fiftyTwoWeekLow}`;
      }
    } catch {}
  }
  const prompts = {
    stock: `請根據即時資料全面分析：\n${enriched}\n\n1.公司簡介\n2.基本面（PE/殖利率/ROE）\n3.技術面走勢\n4.風險\n5.投資建議與目標價`,
    news:  `分析財經新聞投資影響：\n${content}\n\n1.摘要\n2.受影響類股\n3.短期影響\n4.長期影響\n5.應對策略`,
    portfolio:`評估投資組合：\n${content}\n\n1.多元性\n2.集中度風險\n3.預期報酬\n4.風險\n5.優化建議`,
    market:`分析今日台股（${new Date().toLocaleDateString('zh-TW')}）：\n1.大盤走勢\n2.強勢族群\n3.弱勢族群\n4.籌碼動向\n5.明日操作重點\n6.風險提示`,
  };
  try { res.json({ result: await ask(prompts[type] || content, provider) }); }
  catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/technical', async (req, res) => {
  const { stock, price, ma5, ma20, ma60, kd_k, kd_d, rsi, volume, trend, provider } = req.body;
  let histCtx = '';
  try {
    const code = stock.match(/\d{4}/)?.[0];
    if (code) {
      const p1 = new Date(); p1.setDate(p1.getDate() - 30);
      const hist = await yf.historical(code + '.TW', { period1: p1, interval: '1d' }, YFO);
      histCtx = '\n【近5日歷史數據】\n' + hist.slice(-5).map(h =>
        `${h.date.toISOString().split('T')[0]} 收${h.close?.toFixed(0)} 量${h.volume?(h.volume/1e6).toFixed(1)+'M':'—'}`
      ).join('\n');
    }
  } catch {}
  const prompt = `請對「${stock}」進行技術指標分析。\n\n股價：${price} | MA5/20/60：${ma5}/${ma20}/${ma60}\nKD K/D：${kd_k}/${kd_d} | RSI：${rsi}\n成交量：${volume} | 走勢：${trend}${histCtx}\n\n請分析：\n1.均線多空排列\n2.KD指標解讀\n3.RSI強弱評估\n4.支撐與壓力位\n5.短中長期趨勢\n6.操作建議（進場/停損/目標價）`;
  try { res.json({ result: await ask(prompt, provider, 2500) }); }
  catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/allocation', async (req, res) => {
  const { holdings, risk, goal, horizon, provider } = req.body;
  const prompt = `分析並優化投資組合：\n\n${holdings}\n\n風險：${risk} | 目標：${goal} | 期間：${horizon}\n\n請提供：\n1.健診（集中度/多元性/風險評分）\n2.產業/地區配置分析\n3.建議最佳化配置比例\n4.減碼/增碼說明\n5.推薦新標的\n6.預期改善`;
  try { res.json({ result: await ask(prompt, provider, 3000) }); }
  catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/report', async (req, res) => {
  const { portfolio, period, focus, provider } = req.body;
  const prompt = `請為以下投資組合生成「${period}投資分析報告」（${new Date().toLocaleDateString('zh-TW')}）：\n\n${portfolio}\n\n重點：${focus}\n\n一、執行摘要\n二、市場環境回顧\n三、持股績效分析\n四、風險指標評估\n五、重要事件影響\n六、下期展望與策略\n七、調整建議\n八、免責聲明`;
  try { res.json({ result: await ask(prompt, provider, 4096) }); }
  catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/monthly', async (req, res) => {
  const { month, provider } = req.body;
  const prompt = `請生成「${month}台股市場重點摘要」：\n\n## 大盤表現\n## 產業亮點（前3強勢/前3弱勢）\n## 重要財經事件\n## 籌碼動向（外資/投信/自營商）\n## 本月最強個股\n## 下月關鍵觀察指標\n## AI 投資建議重點\n## 風險警示\n\n請生成完整有深度的月報。`;
  try { res.json({ result: await ask(prompt, provider, 3000) }); }
  catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));
app.listen(PORT, () => console.log(`🚀 智投 AI — http://localhost:${PORT}`));