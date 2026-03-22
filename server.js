'use strict';
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

/* ═══════════════════════════════════════════════════════
   AI CLIENTS
═══════════════════════════════════════════════════════ */
const anthropic = process.env.ANTHROPIC_API_KEY
  ? new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY }) : null;
const openai = process.env.OPENAI_API_KEY
  ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;
const FRED_KEY = process.env.FRED_API_KEY || null;

const BASE_SYSTEM = `你是「智投 AI」投資分析系統的一員。請用繁體中文回覆，條理清晰，善用表格與條列式，數據具體。所有分析僅供參考，不構成投資建議。`;

const YFO = { validateResult: false };

/* ═══════════════════════════════════════════════════════
   CACHE
═══════════════════════════════════════════════════════ */
const _cache = new Map();
function getCache(k) {
  const h = _cache.get(k);
  if (h && Date.now() - h.ts < h.ttl) return h.data;
  return null;
}
function setCache(k, data, ttl) { _cache.set(k, { data, ts: Date.now(), ttl }); }

/* ═══════════════════════════════════════════════════════
   AI HELPERS
═══════════════════════════════════════════════════════ */
app.get('/api/models', (_req, res) => res.json({
  claude: !!anthropic, openai: !!openai, fred: !!FRED_KEY,
  default: anthropic ? 'claude' : (openai ? 'openai' : null),
}));

async function ask(prompt, provider = 'claude', maxTokens = 2048, systemOverride = null) {
  const system = systemOverride || BASE_SYSTEM;
  if (provider === 'openai') {
    if (!openai) throw new Error('OpenAI API key 未設定');
    const r = await openai.chat.completions.create({
      model: 'gpt-4o', max_tokens: maxTokens,
      messages: [{ role: 'system', content: system }, { role: 'user', content: prompt }],
    });
    return r.choices[0].message.content;
  }
  if (!anthropic) throw new Error('Anthropic API key 未設定');
  const r = await anthropic.messages.create({
    model: 'claude-opus-4-5', max_tokens: maxTokens, system,
    messages: [{ role: 'user', content: prompt }],
  });
  return r.content[0].text;
}

async function streamAI(messages, provider = 'claude', res, systemOverride = null) {
  const system = systemOverride || BASE_SYSTEM;
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  if (provider === 'openai') {
    if (!openai) throw new Error('OpenAI API key 未設定');
    const s = await openai.chat.completions.create({
      model: 'gpt-4o', max_tokens: 2048, stream: true,
      messages: [{ role: 'system', content: system }, ...messages],
    });
    for await (const c of s) {
      const t = c.choices[0]?.delta?.content || '';
      if (t) res.write(`data: ${JSON.stringify({ text: t })}\n\n`);
    }
  } else {
    if (!anthropic) throw new Error('Anthropic API key 未設定');
    const s = await anthropic.messages.stream({ model: 'claude-opus-4-5', max_tokens: 2048, system, messages });
    for await (const c of s)
      if (c.type === 'content_block_delta' && c.delta.type === 'text_delta')
        res.write(`data: ${JSON.stringify({ text: c.delta.text })}\n\n`);
  }
  res.write('data: [DONE]\n\n');
  res.end();
}

/* ═══════════════════════════════════════════════════════
   QUANTITATIVE UTILITIES
═══════════════════════════════════════════════════════ */

// Rolling statistics (window-based mean + std)
function rollingStats(arr, w) {
  return arr.map((_, i) => {
    if (i < w - 1) return { mean: null, std: null };
    const slice = arr.slice(i - w + 1, i + 1);
    const mean  = slice.reduce((a, b) => a + b, 0) / w;
    const std   = Math.sqrt(slice.reduce((a, b) => a + (b - mean) ** 2, 0) / w);
    return { mean, std };
  });
}

// Z-score anomaly detection (rolling)
function detectAnomalies(values, dates, threshold = 2.5, window = 20) {
  const stats = rollingStats(values, window);
  return stats.reduce((acc, s, i) => {
    if (!s.std || s.std === 0) return acc;
    const z = (values[i] - s.mean) / s.std;
    if (Math.abs(z) >= threshold)
      acc.push({ date: dates[i], value: values[i], zscore: +z.toFixed(2), direction: z > 0 ? 'spike' : 'drop' });
    return acc;
  }, []);
}

// Isolation Forest (lightweight JS implementation)
// Builds random isolation trees and scores anomalies by average path length
function isolationForest(data, nTrees = 100, sampleSize = 256) {
  function iTree(X, depth, maxDepth) {
    if (depth >= maxDepth || X.length <= 1) return { size: X.length };
    const featIdx = Math.floor(Math.random() * X[0].length);
    const vals    = X.map(x => x[featIdx]);
    const min     = Math.min(...vals), max = Math.max(...vals);
    if (min === max) return { size: X.length };
    const split = min + Math.random() * (max - min);
    return {
      featIdx, split,
      left:  iTree(X.filter(x => x[featIdx] < split),  depth + 1, maxDepth),
      right: iTree(X.filter(x => x[featIdx] >= split), depth + 1, maxDepth),
    };
  }
  function pathLen(node, x, depth) {
    if (!node.featIdx && node.featIdx !== 0) {
      const n = node.size;
      return depth + (n <= 1 ? 0 : 2 * (Math.log(n - 1) + 0.5772) - 2 * (n - 1) / n);
    }
    return x[node.featIdx] < node.split
      ? pathLen(node.left,  x, depth + 1)
      : pathLen(node.right, x, depth + 1);
  }
  const maxDepth = Math.ceil(Math.log2(sampleSize));
  const trees = Array.from({ length: nTrees }, () => {
    const sample = [];
    for (let i = 0; i < Math.min(sampleSize, data.length); i++)
      sample.push(data[Math.floor(Math.random() * data.length)]);
    return iTree(sample, 0, maxDepth);
  });
  const c = n => n <= 1 ? 0 : 2 * (Math.log(n - 1) + 0.5772) - 2 * (n - 1) / n;
  return data.map(x => {
    const avgLen = trees.reduce((s, t) => s + pathLen(t, x, 0), 0) / nTrees;
    return 2 ** (-avgLen / c(Math.min(sampleSize, data.length)));
  });
}

// Sharpe Ratio
function calcSharpe(returns, rfRate = 0.02) {
  if (!returns.length) return null;
  const rf  = rfRate / 252;
  const exc = returns.map(r => r - rf);
  const m   = exc.reduce((a, b) => a + b, 0) / exc.length;
  const s   = Math.sqrt(exc.reduce((a, b) => a + (b - m) ** 2, 0) / exc.length);
  return s === 0 ? null : +((m / s) * Math.sqrt(252)).toFixed(3);
}

// Max Drawdown
function calcMaxDrawdown(prices) {
  let peak = prices[0], mdd = 0;
  for (const p of prices) {
    if (p > peak) peak = p;
    mdd = Math.min(mdd, (p - peak) / peak);
  }
  return +((mdd * 100)).toFixed(2);
}

// Calmar Ratio
function calcCalmar(annRet, mdd) {
  return mdd === 0 ? null : +((annRet / Math.abs(mdd))).toFixed(3);
}

// Kelly Criterion: f* = (p*b - q) / b  where b = win/loss ratio
function kellyFraction(winRate, avgWin, avgLoss) {
  if (avgLoss === 0) return 0;
  const b = Math.abs(avgWin / avgLoss);
  const p = winRate / 100;
  const q = 1 - p;
  const f = (p * b - q) / b;
  return Math.max(0, +f.toFixed(4)); // can't be negative
}

/* ═══════════════════════════════════════════════════════
   MARKET DATA ENDPOINTS
═══════════════════════════════════════════════════════ */

const TICKER_LIST = [
  { sym: '2330.TW', label: '台積電'  }, { sym: '2454.TW', label: '聯發科'  },
  { sym: '2317.TW', label: '鴻海'    }, { sym: '^TWII',   label: '台加權'  },
  { sym: '^GSPC',   label: 'S&P500'  }, { sym: '^IXIC',   label: 'NASDAQ'  },
  { sym: '^DJI',    label: '道瓊'    }, { sym: '^VIX',    label: 'VIX'     },
  { sym: 'GC=F',    label: '黃金'    }, { sym: 'CL=F',    label: 'WTI油'   },
  { sym: 'USDTWD=X',label: 'USD/TWD' },
];

/* ── Direct Yahoo Finance v8 chart API (more reliable than quote batch) ── */
async function fetchYFPrice(sym) {
  // Method 1: yf.chart() — uses Yahoo Finance v8 API with crumb (confirmed working on Render)
  try {
    const r = await yf.chart(sym, { range: '5d', interval: '1d', includePrePost: false }, YFO);
    const meta   = r?.meta;
    const quotes = r?.quotes || [];
    // Get price from meta (current price) or last quote
    const price = meta?.regularMarketPrice
      || (quotes.length > 0 ? quotes[quotes.length-1]?.close : null)
      || 0;
    const prev  = meta?.chartPreviousClose || meta?.previousClose || price;
    const change = prev > 0 && price > 0 ? ((price - prev) / prev * 100) : 0;
    if (price > 0) {
      return { price, change, up: change >= 0, symbol: sym, shortName: meta?.shortName || sym };
    }
  } catch (e) {
    console.warn(`fetchYFPrice chart() ${sym}:`, e.message);
  }

  // Method 2: yf.quote() — uses v6/quote API with crumb
  try {
    const q = await yf.quote(sym, {}, YFO);
    const price  = q.regularMarketPrice ?? 0;
    const change = q.regularMarketChangePercent ?? 0;
    if (price > 0) {
      return { price, change, up: change >= 0, symbol: sym, shortName: q.shortName || sym };
    }
  } catch (e) {
    console.warn(`fetchYFPrice quote() ${sym}:`, e.message);
  }

  // Method 3: Direct HTTP with browser headers
  try {
    const enc = encodeURIComponent(sym);
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${enc}?interval=1d&range=5d&includePrePost=false`;
    const r = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'Referer': 'https://finance.yahoo.com/',
      },
      signal: AbortSignal.timeout(8000),
    });
    if (r.ok) {
      const d = await r.json();
      const meta = d?.chart?.result?.[0]?.meta;
      const price = meta?.regularMarketPrice || 0;
      const prev  = meta?.chartPreviousClose || meta?.previousClose || price;
      const change = prev > 0 && price > 0 ? ((price - prev) / prev * 100) : 0;
      if (price > 0) return { price, change, up: change >= 0, symbol: sym, shortName: meta?.shortName || sym };
    }
  } catch (e) {
    console.warn(`fetchYFPrice HTTP ${sym}:`, e.message);
  }

  console.error(`fetchYFPrice ${sym}: all methods failed, returning 0`);
  return { price: 0, change: 0, up: true, symbol: sym, shortName: sym };
}

app.get('/api/ticker', async (_req, res) => {
  const cached = getCache('ticker');
  if (cached) return res.json(cached);
  try {
    // Fetch all in parallel using direct HTTP
    const results = await Promise.all(
      TICKER_LIST.map(async (t) => {
        const q = await fetchYFPrice(t.sym);
        return {
          sym:    t.label,
          ticker: t.sym,
          price:  q.price,
          change: q.change,
          up:     q.up,
        };
      })
    );
    const hasData = results.some(r => r.price > 0);
    console.log('ticker:', results.map(r => `${r.sym}:${r.price}`).join(', '));
    if (hasData) setCache('ticker', results, 60_000);
    res.json(results);
  } catch (e) {
    console.error('ticker:', e.message);
    res.status(500).json({ error: e.message });
  }
});

const WATCHLIST = [
  { sym: '2330.TW', code: '2330', name: '台積電', sector: '半導體'    },
  { sym: '2454.TW', code: '2454', name: '聯發科', sector: 'IC設計'    },
  { sym: '2317.TW', code: '2317', name: '鴻海',   sector: '電子製造'  },
  { sym: '3231.TW', code: '3231', name: '緯創',   sector: 'AI伺服器'  },
  { sym: '2412.TW', code: '2412', name: '中華電', sector: '電信'      },
];

/* ═══════════════════════════════════════════
   DEBUG — inspect raw Yahoo Finance response
═══════════════════════════════════════════ */
app.get('/api/debug', async (req, res) => {
  const sym = req.query.sym || '2330.TW';
  const result = { sym, timestamp: new Date().toISOString() };
  // Test fetchYFPrice (our main method)
  try {
    const fp = await fetchYFPrice(sym);
    result.fetchYFPrice = fp;
  } catch (e) { result.fetchYFPriceError = e.message; }

  try {
    // Test single quote
    const q = await yf.quote(sym, {}, YFO);
    result.quote = {
      symbol:              q.symbol,
      regularMarketPrice:  q.regularMarketPrice,
      regularMarketChange: q.regularMarketChange,
      regularMarketChangePercent: q.regularMarketChangePercent,
      marketState:         q.marketState,
      quoteType:           q.quoteType,
      shortName:           q.shortName,
    };
  } catch (e) { result.quoteError = e.message; }

  try {
    // Test chart endpoint
    const r = await yf.chart(sym, { range: '5d', interval: '1d' }, YFO);
    const quotes = r?.quotes || [];
    result.chart = {
      length: quotes.length,
      firstRow: quotes[0] || null,
      lastRow:  quotes[quotes.length - 1] || null,
    };
  } catch (e) { result.chartError = e.message; }

  try {
    // Test historical
    const p1 = new Date(); p1.setDate(p1.getDate() - 7);
    const h = await yf.historical(sym, { period1: p1, interval: '1d' }, YFO);
    result.historical = {
      length: h?.length || 0,
      firstRow: h?.[0] || null,
    };
  } catch (e) { result.historicalError = e.message; }

  res.json(result);
});

app.get('/api/custom-watchlist', async (req, res) => {
  // Accept ?sym=2330.TW&sym=AAPL&sym=...
  let syms = req.query.sym;
  if (!syms) return res.json([]);
  if (!Array.isArray(syms)) syms = [syms];

  // Normalize: 4-digit → add .TW, already has dot → keep, else keep as-is
  syms = syms.slice(0, 5).map(s => {
    s = s.toUpperCase().trim();
    if (/^\d{4}$/.test(s)) return s + '.TW';
    return s;
  });

  try {
    const quotes = await yf.quote(syms, {}, YFO);
    const arr    = Array.isArray(quotes) ? quotes : [quotes];
    const result = await Promise.all(syms.map(async (sym) => {
      const code = sym.replace('.TW','');
      const q = await fetchYFPrice(sym);
      const price  = q.price  > 0 ? q.price  : null;
      const change = q.price  > 0 ? q.change : null;
      let name = code, sector = code, pe = null, mktCap = null, week52H = null, week52L = null;
      try {
        const detail = await yf.quote(sym, {}, YFO);
        name    = detail.shortName || detail.longName || code;
        sector  = detail.sector    || code;
        pe      = detail.trailingPE       ?? null;
        mktCap  = detail.marketCap        ?? null;
        week52H = detail.fiftyTwoWeekHigh ?? null;
        week52L = detail.fiftyTwoWeekLow  ?? null;
      } catch {}
      return { sym, code, name, sector, price, change, up: change != null ? change >= 0 : true, pe, mktCap, week52H, week52L };
    }));
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get('/api/watchlist', async (_req, res) => {
  const cached = getCache('watchlist');
  if (cached) return res.json(cached);
  try {
    // Use direct HTTP for price + yf library for fundamentals
    const result = await Promise.all(WATCHLIST.map(async (w) => {
      const q = await fetchYFPrice(w.sym);
      const price  = q.price  > 0 ? q.price  : null;
      const change = q.price  > 0 ? q.change : null;
      // Try to get PE from yf library (non-critical)
      let pe = null, mktCap = null, week52H = null, week52L = null;
      try {
        const detail = await yf.quote(w.sym, {}, YFO);
        pe      = detail.trailingPE       ?? null;
        mktCap  = detail.marketCap        ?? null;
        week52H = detail.fiftyTwoWeekHigh ?? null;
        week52L = detail.fiftyTwoWeekLow  ?? null;
      } catch {}
      console.log(`watchlist ${w.sym}: price=${price}`);
      return { ...w, price, change, up: change != null ? change >= 0 : true, pe, mktCap, week52H, week52L };
    }));

    const anyPrice = result.some(r => r.price != null && r.price > 0);
    if (anyPrice) setCache('watchlist', result, 60_000);
    console.log('watchlist:', result.map(r => `${r.code}:${r.price}`).join(', '));
    res.json(result);
  } catch (e) {
    console.error('watchlist:', e.message);
    res.status(500).json({ error: e.message });
  }
});

app.get('/api/chart/:symbol', async (req, res) => {
  const { symbol } = req.params;
  const period   = req.query.period   || '6mo';
  const interval = req.query.interval || '1d';
  const k = `chart:${symbol}:${period}:${interval}`;
  const cached = getCache(k);
  if (cached) return res.json(cached);

  // Map period string to yahoo-finance2 range
  const rangeMap = {
    '1mo':'1mo', '3mo':'3mo', '6mo':'6mo',
    '1y':'1y',   '2y':'2y',   '5y':'5y',
  };
  const range = rangeMap[period] || '6mo';

  // Map interval
  const intervalMap = {
    '1d':'1d', '1wk':'1wk', '1mo':'1mo',
  };
  const intv = intervalMap[interval] || '1d';

  async function tryFetch(attempts = 3) {
    for (let i = 0; i < attempts; i++) {
      try {
        // Try yf.chart() first (v8 API — more reliable than v7 CSV download)
        const r = await yf.chart(symbol, {
          range,
          interval: intv,
          includePrePost: false,
        }, YFO);

        const quotes = r?.quotes || r?.indicators?.quote?.[0] || [];
        const timestamps = r?.timestamp || [];

        let rows = [];
        if (Array.isArray(quotes) && quotes.length > 0 && quotes[0]?.close != null) {
          // chart() returns array of {date, open, high, low, close, volume}
          rows = quotes
            .filter(q => q.close != null)
            .map(q => ({
              date:   q.date instanceof Date
                        ? q.date.toISOString().split('T')[0]
                        : new Date(q.date * 1000).toISOString().split('T')[0],
              open:   q.open   ?? null,
              high:   q.high   ?? null,
              low:    q.low    ?? null,
              close:  q.close,
              volume: q.volume ?? null,
            }));
        } else if (timestamps.length > 0 && r?.indicators?.quote?.[0]) {
          // Alternative structure: parallel arrays
          const q = r.indicators.quote[0];
          rows = timestamps
            .map((ts, idx) => ({
              date:   new Date(ts * 1000).toISOString().split('T')[0],
              open:   q.open?.[idx]   ?? null,
              high:   q.high?.[idx]   ?? null,
              low:    q.low?.[idx]    ?? null,
              close:  q.close?.[idx]  ?? null,
              volume: q.volume?.[idx] ?? null,
            }))
            .filter(d => d.close != null);
        }

        if (rows.length > 0) return rows;

        // yf.chart() returned empty — fall back to yf.historical()
        console.log(`chart() returned empty for ${symbol}, trying historical()...`);
        const dayMap = {'1mo':30,'3mo':90,'6mo':180,'1y':365,'2y':730,'5y':1825};
        const p1 = new Date();
        p1.setDate(p1.getDate() - (dayMap[period] || 180));
        const hist = await yf.historical(symbol, { period1: p1, interval: intv }, YFO);
        const histRows = (hist || [])
          .map(d => ({
            date:   d.date instanceof Date ? d.date.toISOString().split('T')[0] : String(d.date).split('T')[0],
            open:   d.open   ?? null,
            high:   d.high   ?? null,
            low:    d.low    ?? null,
            close:  d.close  ?? null,
            volume: d.volume ?? null,
          }))
          .filter(d => d.close != null);

        if (histRows.length > 0) return histRows;

        if (i < attempts - 1) {
          await new Promise(r => setTimeout(r, 1500 * (i + 1)));
        }
      } catch (err) {
        console.warn(`chart attempt ${i+1} for ${symbol}:`, err.message);
        if (i === attempts - 1) throw err;
        await new Promise(r => setTimeout(r, 1500 * (i + 1)));
      }
    }
    return [];
  }

  try {
    const result = await tryFetch(3);
    if (result.length > 0) {
      setCache(k, result, 300_000);
      console.log(`chart ${symbol}: ${result.length} rows OK`);
    } else {
      console.warn(`chart ${symbol}: returned 0 rows after all attempts`);
    }
    res.json(result);
  } catch (e) {
    console.error(`chart ${symbol} failed:`, e.message);
    res.status(500).json({ error: e.message });
  }
});

app.get('/api/twse/market', async (_req, res) => {
  const cached = getCache('twse:market');
  if (cached) return res.json(cached);
  try {
    const r = await fetch('https://www.twse.com.tw/rwd/zh/fund/TWT38U?response=json&selectType=All');
    const result = { fii: null, fetched: new Date().toISOString() };
    if (r.ok) {
      const d = await r.json();
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
      yf.quoteSummary(sym, { modules: ['summaryDetail','defaultKeyStatistics','financialData','assetProfile','earnings'] }, YFO),
    ]);
    const q = qR.status === 'fulfilled' ? qR.value : {};
    const s = sR.status === 'fulfilled' ? sR.value : {};
    const result = {
      symbol: sym, name: q.shortName || q.longName || sym,
      price: q.regularMarketPrice, change: q.regularMarketChange,
      changePct: q.regularMarketChangePercent, volume: q.regularMarketVolume,
      mktCap: q.marketCap, pe: q.trailingPE ?? s.summaryDetail?.trailingPE,
      pb: q.priceToBook ?? s.defaultKeyStatistics?.priceToBook,
      eps: q.epsTrailingTwelveMonths,
      divYield: q.dividendYield ?? s.summaryDetail?.dividendYield,
      roe: s.financialData?.returnOnEquity,
      fcf: s.financialData?.freeCashflow,
      grossMargins: s.financialData?.grossMargins,
      operatingMargins: s.financialData?.operatingMargins,
      revenueGrowth: s.financialData?.revenueGrowth,
      debtToEquity: s.financialData?.debtToEquity,
      currentRatio: s.financialData?.currentRatio,
      week52H: q.fiftyTwoWeekHigh, week52L: q.fiftyTwoWeekLow,
      industry: s.assetProfile?.industry, sector: s.assetProfile?.sector,
      desc: s.assetProfile?.longBusinessSummary,
    };
    setCache('stock:' + sym, result, 120_000);
    res.json(result);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

/* ── News RSS ─────────────────────────────────────── */
app.get('/api/news/:symbol', async (req, res) => {
  const sym = req.params.symbol;
  const cached = getCache('news:' + sym);
  if (cached) return res.json(cached);
  try {
    // Yahoo Finance RSS
    const encoded = encodeURIComponent(sym.includes('.TW') ? sym : sym + '.TW');
    const url = `https://feeds.finance.yahoo.com/rss/2.0/headline?s=${encoded}&region=TW&lang=zh-TW`;
    const r   = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    const xml = await r.text();

    // Parse RSS items with regex (no xml library needed)
    const items = [];
    const re = /<item>([\s\S]*?)<\/item>/g;
    let m;
    while ((m = re.exec(xml)) !== null && items.length < 10) {
      const item  = m[1];
      const title = (/<title><!\[CDATA\[(.*?)\]\]>/.exec(item) || /<title>(.*?)<\/title>/.exec(item) || [])[1] || '';
      const link  = (/<link>(.*?)<\/link>/.exec(item) || [])[1] || '';
      const pubDate = (/<pubDate>(.*?)<\/pubDate>/.exec(item) || [])[1] || '';
      if (title) items.push({ title: title.trim(), link: link.trim(), pubDate: pubDate.trim() });
    }

    setCache('news:' + sym, items, 600_000); // 10 min
    res.json(items);
  } catch (e) {
    res.json([]); // return empty gracefully
  }
});

/* ═══════════════════════════════════════════════════════
   MACRO ECONOMIC DATA
═══════════════════════════════════════════════════════ */
async function fetchFRED(seriesId, limit = 3) {
  if (!FRED_KEY) return null;
  try {
    const url = `https://api.stlouisfed.org/fred/series/observations?series_id=${seriesId}&api_key=${FRED_KEY}&file_type=json&limit=${limit}&sort_order=desc`;
    const d   = await fetch(url).then(r => r.json());
    return d.observations?.filter(o => o.value !== '.').slice(0, limit) || null;
  } catch { return null; }
}

async function fetchNFP() {
  try {
    const year = new Date().getFullYear();
    const r = await fetch('https://api.bls.gov/publicAPI/v2/timeseries/data/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ seriesid: ['CES0000000001'], startyear: String(year-1), endyear: String(year) }),
    });
    const d = await r.json();
    return (d.Results?.series?.[0]?.data || []).slice(0, 3).map(s => ({
      date: `${s.year}-${s.period.replace('M','')}`, value: s.value,
    }));
  } catch { return null; }
}

app.get('/api/macro', async (_req, res) => {
  const cached = getCache('macro');
  if (cached) return res.json(cached);
  try {
    const [fedRate, cpi, pce, unemployment, vixQ, treasury10y, nfp] = await Promise.allSettled([
      fetchFRED('FEDFUNDS', 3), fetchFRED('CPIAUCSL', 3), fetchFRED('PCEPI', 3),
      fetchFRED('UNRATE', 3),   yf.quote('^VIX', {}, YFO), fetchFRED('DGS10', 3), fetchNFP(),
    ]);
    const result = {
      fedRate:      fedRate.status === 'fulfilled'      ? fedRate.value      : null,
      cpi:          cpi.status === 'fulfilled'          ? cpi.value          : null,
      pce:          pce.status === 'fulfilled'          ? pce.value          : null,
      unemployment: unemployment.status === 'fulfilled' ? unemployment.value : null,
      vix:          vixQ.status === 'fulfilled'         ? { value: vixQ.value.regularMarketPrice, change: vixQ.value.regularMarketChangePercent } : null,
      treasury10y:  treasury10y.status === 'fulfilled'  ? treasury10y.value  : null,
      nfp:          nfp.status === 'fulfilled'          ? nfp.value          : null,
      fredAvailable: !!FRED_KEY, fetched: new Date().toISOString(),
    };
    setCache('macro', result, 3_600_000);
    res.json(result);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/macro/analyze', async (req, res) => {
  const { macroData, provider } = req.body;
  const lines = [
    macroData.fedRate?.[0]      ? `聯邦基金利率：${macroData.fedRate[0].value}%（前次：${macroData.fedRate[1]?.value||'—'}%）` : '聯邦基金利率：未提供（需 FRED_API_KEY）',
    macroData.cpi?.[0]          ? `CPI：${macroData.cpi[0].value}（${macroData.cpi[0].date}）` : 'CPI：未提供',
    macroData.unemployment?.[0] ? `失業率：${macroData.unemployment[0].value}%` : '失業率：未提供',
    macroData.vix               ? `VIX：${macroData.vix.value?.toFixed(2)}（${macroData.vix.change?.toFixed(2)}%）` : 'VIX：未提供',
    macroData.nfp?.[0]          ? `非農就業：${macroData.nfp[0].value}千人（${macroData.nfp[0].date}）` : '非農：未提供',
    macroData.treasury10y?.[0]  ? `10年美債：${macroData.treasury10y[0].value}%` : '10年美債：未提供',
  ].join('\n');

  const prompt = `請根據以下總經數據進行宏觀環境分析：\n\n${lines}\n\n請提供：\n一、經濟週期判斷（擴張/過熱/收縮/復甦）\n二、聯準會政策展望（升息/暫停/降息概率）\n三、通膨與就業趨勢\n四、VIX 市場恐慌解讀\n五、對台股科技類股的具體影響\n六、建議資產配置方向（股/債/黃金/現金比例）\n七、未來3個月關鍵風險事件`;
  try { res.json({ result: await ask(prompt, provider, 3000) }); }
  catch (e) { res.status(500).json({ error: e.message }); }
});

/* ═══════════════════════════════════════════════════════
   NEW: 7 ANALYST REPORTS
   Each analyst has a distinct persona and analysis angle
═══════════════════════════════════════════════════════ */
const ANALYSTS = [
  {
    id: 'fundamental',
    name: '基本面分析師',
    system: `你是一位嚴格的基本面分析師，專注於財務報表解讀。你只相信數字，對美化話術保持懷疑。擅長：財務三率（毛利率/營益率/淨利率）、負債比、現金流健康度、EPS 成長軌跡。請用繁體中文，以表格呈現數字，結論要明確。`,
    promptFn: (stock, data) => `請對「${stock}」進行基本面深度分析。\n\n${data}\n\n分析重點：\n1. 財務三率趨勢（毛利率/營業利益率/淨利率）\n2. 負債比與流動比率（財務安全性）\n3. 自由現金流（FCF）健康度\n4. EPS 成長軌跡（加速/減速？）\n5. ROE 趨勢（股東權益報酬率）\n6. 財務評分（1-10分）與投資評等`,
  },
  {
    id: 'technical',
    name: '技術分析師',
    system: `你是一位精通圖形分析的技術分析師，信奉「價量是市場的語言」。擅長：均線系統、KD/RSI/MACD、支撐壓力位、型態識別（頭肩頂/底、旗形、杯柄）。請用繁體中文，以清晰的多空論述呈現。`,
    promptFn: (stock, data) => `請對「${stock}」進行技術面分析。\n\n${data}\n\n分析重點：\n1. 均線多空排列（月/週/日線）\n2. 量價關係（量能是否配合？）\n3. KD/RSI 超買超賣訊號\n4. 關鍵支撐位與壓力位\n5. 目前K線型態辨識\n6. 建議進場點、停損點、目標價`,
  },
  {
    id: 'news',
    name: '新聞情緒分析師',
    system: `你是一位媒體與市場情緒專家，擅長從新聞標題提取關鍵訊號，判斷消息面對股價的短期衝擊力。你了解「利多出盡」和「利空不跌」的市場規律。請用繁體中文，評估新聞的情緒偏向與影響時效。`,
    promptFn: (stock, news) => `請分析以下「${stock}」相關新聞的市場情緒與股價衝擊：\n\n${news}\n\n分析重點：\n1. 關鍵字情緒評分（正面/負面/中性，-10到+10）\n2. 新聞衝擊力（高/中/低）與持續時效\n3. 是否有「利多出盡」或「利空反彈」的逆向訊號\n4. 主力法人可能的解讀方式\n5. 短期1-5個交易日的預期影響`,
  },
  {
    id: 'sentiment',
    name: '市場情緒分析師',
    system: `你是一位擅長解讀群眾心理與市場情緒的分析師，了解散戶行為偏差、恐慌貪婪週期、從眾效應對股價的影響。請用繁體中文，結合 VIX 與技術指標判斷市場情緒極值。`,
    promptFn: (stock, data) => `請對「${stock}」進行市場情緒面分析。\n\n${data}\n\n分析重點：\n1. 目前市場對此股的情緒偏向（過度樂觀/恐慌/理性）\n2. VIX 與大盤情緒對此股的連動影響\n3. 法人（外資/投信）近期籌碼動向解讀\n4. 融資融券變化的多空含義\n5. 情緒指標是否達到極值（反向訊號？）\n6. 情緒面評分（1-10）與操作建議`,
  },
  {
    id: 'longterm',
    name: '長線投資策略師',
    system: `你是一位信奉價值投資的長線策略師，以巴菲特/費雪的框架評估企業護城河與長期競爭優勢。你關注5-10年的產業趨勢、企業定價能力、管理層品質。請用繁體中文，給出長期持有的是非題。`,
    promptFn: (stock, data) => `請對「${stock}」進行長線投資價值評估。\n\n${data}\n\n分析重點：\n1. 企業護城河強度（品牌/技術/網路效應/成本優勢/轉換成本）\n2. 產業未來5-10年趨勢（順風/逆風？）\n3. 企業定價能力（能否轉嫁成本？）\n4. 管理層資本配置能力（ROE/買回庫藏股/現金股利）\n5. 長線風險（顛覆性競爭/政治風險/技術過時）\n6. 值得長期持有嗎？（是/否，並說明理由）`,
  },
  {
    id: 'trader',
    name: '短線交易員',
    system: `你是一位積極的短線交易員，專注於3-20個交易日的波段操作。你只關心「現在能不能賺錢」，不在意長期故事。擅長識別催化劑（財報/法說/訂單/政策）、量能異動、籌碼集中度變化。請用繁體中文，給出直接的交易建議。`,
    promptFn: (stock, data) => `請對「${stock}」提供短線交易計劃（3-20個交易日波段）。\n\n${data}\n\n交易計劃需包含：\n1. 目前適合做多還是做空（或觀望）？理由？\n2. 進場點位（最佳買點區間）\n3. 停損點位（跌破即出場）\n4. 第一目標價（保守獲利點）\n5. 第二目標價（積極獲利點）\n6. 預期持倉時間\n7. 需要注意的近期催化劑（財報日/法說會/重要事件）`,
  },
  {
    id: 'final',
    name: '最終決策官',
    system: `你是投資委員會的最終決策官，負責綜合所有分析師的意見，做出最終投資評等。你特別注意各分析師之間的分歧點，並對矛盾之處進行裁決。請用繁體中文，給出清晰的「強力買入/買入/持有/減持/強力賣出」結論，並附上信心指數（1-10）。`,
    promptFn: (stock, allReports) => `請綜合以下7位分析師的報告，對「${stock}」給出最終投資決策：\n\n${allReports}\n\n最終報告需包含：\n1. 執行摘要（3行以內）\n2. 各分析師觀點共識點\n3. 各分析師觀點分歧點（重點裁決）\n4. 最終投資評等：強力買入 / 買入 / 持有 / 減持 / 強力賣出\n5. 信心指數（1-10分）\n6. 目標價區間（12個月）\n7. 最大風險因素（1-3個）\n8. 觸發升評或降評的條件`,
  },
];

app.post('/api/seven-analysts', async (req, res) => {
  const { stock, provider } = req.body;
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const send = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`);

  try {
    // Step 1: Fetch comprehensive stock data
    send({ type: 'status', msg: '正在抓取即時市場數據...' });
    const code = stock.match(/\d{4}/)?.[0];
    const sym  = code ? code + '.TW' : stock;

    let stockData = `股票：${stock}，資料抓取中...`;
    let newsData  = '新聞資料抓取中...';
    let p1For5yr  = new Date(); p1For5yr.setFullYear(p1For5yr.getFullYear() - 5);

    const [qR, sR, histR, newsR] = await Promise.allSettled([
      yf.quote(sym, {}, YFO),
      yf.quoteSummary(sym, { modules: ['summaryDetail','defaultKeyStatistics','financialData','assetProfile'] }, YFO),
      yf.historical(sym, { period1: p1For5yr, interval: '1mo' }, YFO),
      fetch(`https://feeds.finance.yahoo.com/rss/2.0/headline?s=${encodeURIComponent(sym)}&region=TW&lang=zh-TW`, { headers: { 'User-Agent': 'Mozilla/5.0' } }).then(r => r.text()),
    ]);

    if (qR.status === 'fulfilled') {
      const q = qR.value, s = sR.status === 'fulfilled' ? sR.value : {};
      stockData = `【即時行情】
股價：${q.regularMarketPrice} | 漲跌：${q.regularMarketChangePercent?.toFixed(2)}%
52週高低：${q.fiftyTwoWeekHigh} / ${q.fiftyTwoWeekLow}
本益比：${q.trailingPE?.toFixed(1)||'—'} | 市值：${q.marketCap?(q.marketCap/1e8).toFixed(0)+'億':'—'}
EPS：${q.epsTrailingTwelveMonths?.toFixed(2)||'—'} | 殖利率：${q.dividendYield?(q.dividendYield*100).toFixed(2)+'%':'—'}
【基本面】
毛利率：${s.financialData?.grossMargins?(s.financialData.grossMargins*100).toFixed(1)+'%':'—'}
營業利益率：${s.financialData?.operatingMargins?(s.financialData.operatingMargins*100).toFixed(1)+'%':'—'}
ROE：${s.financialData?.returnOnEquity?(s.financialData.returnOnEquity*100).toFixed(1)+'%':'—'}
負債比：${s.financialData?.debtToEquity?.toFixed(1)||'—'}
流動比率：${s.financialData?.currentRatio?.toFixed(2)||'—'}
自由現金流：${s.financialData?.freeCashflow?(s.financialData.freeCashflow/1e8).toFixed(1)+'億':'—'}
營收成長率：${s.financialData?.revenueGrowth?(s.financialData.revenueGrowth*100).toFixed(1)+'%':'—'}
產業：${s.assetProfile?.industry||'—'}`;
    }

    // Parse news
    if (newsR.status === 'fulfilled') {
      const xml = newsR.value;
      const re  = /<item>([\s\S]*?)<\/item>/g;
      const headlines = [];
      let m;
      while ((m = re.exec(xml)) !== null && headlines.length < 8) {
        const t = (/<title><!\[CDATA\[(.*?)\]\]>/.exec(m[1]) || /<title>(.*?)<\/title>/.exec(m[1]) || [])[1];
        if (t) headlines.push(t.trim());
      }
      newsData = headlines.length ? headlines.join('\n') : '暫無最新新聞';
    }

    // Historical prices for technical context
    let techData = stockData;
    if (histR.status === 'fulfilled' && histR.value.length > 0) {
      const hist  = histR.value;
      const last6 = hist.slice(-6).map(h => `${h.date.toISOString().split('T')[0]} 收盤:${h.close?.toFixed(0)}`).join(' | ');
      techData = stockData + `\n【近6個月月線收盤】\n${last6}`;
    }

    // Step 2: Run analysts 1-6 sequentially (except final)
    const reports = {};
    const nonFinal = ANALYSTS.filter(a => a.id !== 'final');

    for (const analyst of nonFinal) {
      send({ type: 'status', msg: `${analyst.name} 正在分析...` });
      const inputData = analyst.id === 'news' || analyst.id === 'sentiment'
        ? `${stockData}\n\n【最新新聞標題】\n${newsData}`
        : techData;
      try {
        const result = await ask(analyst.promptFn(stock, inputData), provider, 1500, analyst.system);
        reports[analyst.id] = result;
        send({ type: 'analyst', id: analyst.id, name: analyst.name, report: result });
      } catch (e) {
        reports[analyst.id] = `分析失敗：${e.message}`;
        send({ type: 'analyst', id: analyst.id, name: analyst.name, report: reports[analyst.id] });
      }
    }

    // Step 3: Final decision
    send({ type: 'status', msg: '最終決策官正在綜合所有報告...' });
    const final = ANALYSTS.find(a => a.id === 'final');
    const allReportsText = Object.entries(reports).map(([id, r]) => {
      const a = ANALYSTS.find(x => x.id === id);
      return `【${a?.name || id}】\n${r}`;
    }).join('\n\n---\n\n');

    const finalReport = await ask(final.promptFn(stock, allReportsText), provider, 2000, final.system);
    send({ type: 'analyst', id: 'final', name: final.name, report: finalReport });
    send({ type: 'done' });
  } catch (e) {
    send({ type: 'error', msg: e.message });
  }
  res.end();
});

/* ═══════════════════════════════════════════════════════
   NEW: BULL vs BEAR DEBATE
═══════════════════════════════════════════════════════ */
app.post('/api/debate', async (req, res) => {
  const { stock, provider } = req.body;
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const send = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`);

  try {
    // Fetch stock data
    const code = stock.match(/\d{4}/)?.[0];
    const sym  = code ? code + '.TW' : stock;
    let stockCtx = `股票：${stock}`;

    const [qR, sR] = await Promise.allSettled([
      yf.quote(sym, {}, YFO),
      yf.quoteSummary(sym, { modules: ['financialData','summaryDetail'] }, YFO),
    ]);
    if (qR.status === 'fulfilled') {
      const q = qR.value, s = sR.status === 'fulfilled' ? sR.value : {};
      stockCtx = `股票：${stock}\n股價：${q.regularMarketPrice} | PE：${q.trailingPE?.toFixed(1)||'—'} | 殖利率：${q.dividendYield?(q.dividendYield*100).toFixed(2)+'%':'—'}\nROE：${s.financialData?.returnOnEquity?(s.financialData.returnOnEquity*100).toFixed(1)+'%':'—'} | FCF：${s.financialData?.freeCashflow?(s.financialData.freeCashflow/1e8).toFixed(1)+'億':'—'}\n52週高低：${q.fiftyTwoWeekHigh} / ${q.fiftyTwoWeekLow}`;
    }

    const BULL_SYSTEM = `你是一位樂觀的多頭分析師。你的任務是找出這檔股票所有值得買入的理由，並為看多立場進行最強力的辯護。你必須承認風險存在，但要說明為何這些風險可以克服。請用繁體中文，用有說服力的方式呈現。`;
    const BEAR_SYSTEM = `你是一位悲觀的空頭分析師。你的任務是找出這檔股票所有應該避開的理由，並為看空立場進行最強力的辯護。你必須承認有利因素存在，但要說明為何這些優勢可能被高估。請用繁體中文，用有說服力的方式呈現。`;
    const JUDGE_SYSTEM = `你是一位中立的投資裁判，負責評估多空雙方的論點，找出誰的論述更有說服力，更有實質數據支撐。你的結論不一定要偏向任何一方，但要指出哪些論點最關鍵。請用繁體中文，客觀公正地裁決。`;

    // Round 1: Opening arguments
    send({ type: 'status', msg: '多頭分析師提出開場論點...' });
    const bullOpen = await ask(
      `請針對「${stockCtx}」提出你的多頭開場論點（3-5個核心理由，每個100字以內）：`,
      provider, 800, BULL_SYSTEM
    );
    send({ type: 'round', round: 1, side: 'bull', label: '多頭開場', content: bullOpen });

    send({ type: 'status', msg: '空頭分析師提出反駁...' });
    const bearOpen = await ask(
      `多頭分析師剛說：\n「${bullOpen}」\n\n請針對「${stockCtx}」提出你的空頭反駁論點：`,
      provider, 800, BEAR_SYSTEM
    );
    send({ type: 'round', round: 1, side: 'bear', label: '空頭反駁', content: bearOpen });

    // Round 2: Rebuttal
    send({ type: 'status', msg: '多頭進行第二輪反擊...' });
    const bullRebuttal = await ask(
      `空頭分析師剛說：\n「${bearOpen}」\n\n請針對這些空頭論點逐一反擊，並補充你最強的多頭論據：`,
      provider, 800, BULL_SYSTEM
    );
    send({ type: 'round', round: 2, side: 'bull', label: '多頭反擊', content: bullRebuttal });

    send({ type: 'status', msg: '空頭進行最後論述...' });
    const bearFinal = await ask(
      `多頭分析師反擊說：\n「${bullRebuttal}」\n\n請給出你最終的空頭結論，強調最關鍵的風險點：`,
      provider, 800, BEAR_SYSTEM
    );
    send({ type: 'round', round: 2, side: 'bear', label: '空頭總結', content: bearFinal });

    // Judge verdict
    send({ type: 'status', msg: '裁判進行最終裁決...' });
    const judgePrompt = `請評估以下多空辯論，並給出裁決。

【股票背景】
${stockCtx}

【多頭論點】
${bullOpen}

【空頭論點】
${bearOpen}

【多頭反擊】
${bullRebuttal}

【空頭總結】
${bearFinal}

裁決需包含：
1. 哪方論點更有說服力？為什麼？
2. 最關鍵的3個爭議點
3. 中立的投資建議
4. 若你必須選邊，目前偏多還是偏空？信心指數多少？`;

    const verdict = await ask(judgePrompt, provider, 1200, JUDGE_SYSTEM);
    send({ type: 'verdict', content: verdict });
    send({ type: 'done' });
  } catch (e) {
    send({ type: 'error', msg: e.message });
  }
  res.end();
});

/* ═══════════════════════════════════════════════════════
   ENHANCED VALUATION: DCF 2.0 + PE BAND (Std Dev)
═══════════════════════════════════════════════════════ */
app.post('/api/valuation', async (req, res) => {
  const { stock, price, pe, pb, roe, eps, growth, sector, provider } = req.body;

  let realData = `用戶提供：股價${price}、PE${pe}、PB${pb}、ROE${roe}%、EPS${eps}、成長率${growth}%`;
  let dcfBlock = '', peBandBlock = '', historyBlock = '';

  try {
    const code = stock.match(/\d{4}/)?.[0];
    const sym  = code ? code + '.TW' : stock;

    const [qR, sR, histR] = await Promise.allSettled([
      yf.quote(sym, {}, YFO),
      yf.quoteSummary(sym, { modules: ['financialData','summaryDetail','defaultKeyStatistics'] }, YFO),
      yf.historical(sym, { period1: (() => { const d = new Date(); d.setFullYear(d.getFullYear()-5); return d; })(), interval: '1mo' }, YFO),
    ]);

    const q = qR.status === 'fulfilled' ? qR.value : {};
    const s = sR.status === 'fulfilled' ? sR.value : {};
    const currentPrice = q.regularMarketPrice || parseFloat(price) || 0;
    const currentEPS   = q.epsTrailingTwelveMonths || parseFloat(eps) || 0;

    realData = `【Yahoo Finance 即時資料】
股價：${currentPrice} | PE：${q.trailingPE?.toFixed(1)||pe||'—'} | EPS：${currentEPS.toFixed(2)||'—'}
殖利率：${q.dividendYield?(q.dividendYield*100).toFixed(2)+'%':'—'} | ROE：${s.financialData?.returnOnEquity?(s.financialData.returnOnEquity*100).toFixed(1)+'%':roe+'%'}
市值：${q.marketCap?(q.marketCap/1e8).toFixed(0)+'億':'—'} | 52週高低：${q.fiftyTwoWeekHigh} / ${q.fiftyTwoWeekLow}
FCF：${s.financialData?.freeCashflow?(s.financialData.freeCashflow/1e8).toFixed(1)+'億':'—'}
毛利率：${s.financialData?.grossMargins?(s.financialData.grossMargins*100).toFixed(1)+'%':'—'}
營收成長：${s.financialData?.revenueGrowth?(s.financialData.revenueGrowth*100).toFixed(1)+'%':'—'}`;

    // ── PE Band with standard deviation ──
    if (histR.status === 'fulfilled' && histR.value.length > 10 && currentEPS > 0) {
      const hist   = histR.value;
      const prices = hist.map(h => h.close).filter(Boolean);
      const peVals = prices.map(p => p / currentEPS).filter(v => v > 0 && v < 200); // remove outliers

      if (peVals.length > 5) {
        const peMean = peVals.reduce((a,b) => a+b, 0) / peVals.length;
        const peStd  = Math.sqrt(peVals.reduce((a,b) => a + (b-peMean)**2, 0) / peVals.length);
        const peMin  = Math.min(...peVals), peMax = Math.max(...peVals);
        const currentPE = q.trailingPE || parseFloat(pe) || 0;

        // Price targets at different PE bands
        const p1sd_lo = currentEPS * (peMean - peStd);
        const p_mean  = currentEPS * peMean;
        const p1sd_hi = currentEPS * (peMean + peStd);
        const p2sd_hi = currentEPS * (peMean + 2*peStd);

        const percentile = ((currentPrice - (currentEPS * peMin)) /
                            ((currentEPS * peMax) - (currentEPS * peMin)) * 100).toFixed(1);

        peBandBlock = `\n【PE Band 分析（5年標準差方法）】
歷史 PE 均值：${peMean.toFixed(1)} | 標準差：${peStd.toFixed(1)}
PE 區間：${peMin.toFixed(1)} ~ ${peMax.toFixed(1)}
目前 PE：${currentPE.toFixed(1)} → 位於5年歷史 ${percentile}% 分位

PE Band 對應股價（依 EPS ${currentEPS.toFixed(2)} 計算）：
-2σ (超低估)：${(currentEPS*(peMean-2*peStd)).toFixed(0)} 元
-1σ (低估)：${p1sd_lo.toFixed(0)} 元
均值 (合理)：${p_mean.toFixed(0)} 元
+1σ (偏高)：${p1sd_hi.toFixed(0)} 元
+2σ (超高估)：${p2sd_hi.toFixed(0)} 元
目前股價 ${currentPrice} 位於：${
  currentPrice < p1sd_lo ? '低估區間（-1σ以下）' :
  currentPrice < p_mean  ? '相對合理偏低' :
  currentPrice < p1sd_hi ? '相對合理偏高' :
  currentPrice < p2sd_hi ? '偏高區間（+1σ~+2σ）' : '超高估區間（+2σ以上）'
}`;
      }

      // ── DCF 2.0 ──
      const fcf = s.financialData?.freeCashflow;
      if (fcf && fcf > 0) {
        const g     = parseFloat(growth) / 100 || 0.10;
        // WACC estimation (simplified)
        const debtRatio = s.financialData?.debtToEquity
          ? (s.financialData.debtToEquity / (s.financialData.debtToEquity + 100))
          : 0.3;
        const equityCost = 0.12; // assumed cost of equity
        const debtCost   = 0.04; // assumed after-tax cost of debt
        const wacc = equityCost * (1 - debtRatio) + debtCost * debtRatio;
        const terminalG = Math.min(0.03, g * 0.25); // terminal growth = 25% of initial, max 3%

        let dcfTotal = 0, projFCF = fcf;
        const yearRows = [];
        for (let yr = 1; yr <= 5; yr++) {
          const decayG = g * Math.pow(0.85, yr - 1); // growth decays 15% per year
          projFCF = projFCF * (1 + Math.max(decayG, terminalG));
          const pv = projFCF / Math.pow(1 + wacc, yr);
          dcfTotal += pv;
          yearRows.push(`第${yr}年 FCF預估：${(projFCF/1e8).toFixed(1)}億 → 現值：${(pv/1e8).toFixed(1)}億 (成長率 ${(Math.max(decayG, terminalG)*100).toFixed(1)}%)`);
        }
        const tv   = projFCF * (1 + terminalG) / (wacc - terminalG);
        const tvPV = tv / Math.pow(1 + wacc, 5);
        dcfTotal  += tvPV;

        const shares = q.marketCap ? q.marketCap / currentPrice : null;
        const dcfPS  = shares ? dcfTotal / shares : null;

        dcfBlock = `\n【DCF 2.0 現金流折現模型】
估算 WACC：${(wacc*100).toFixed(1)}% (股權成本 ${(equityCost*100)}%，負債比 ${(debtRatio*100).toFixed(0)}%)
用戶輸入初始成長率：${growth}%（逐年遞減15%）
終值永續成長率：${(terminalG*100).toFixed(1)}%

${yearRows.join('\n')}
終值（第5年後）：${(tv/1e8).toFixed(0)}億 → 現值：${(tvPV/1e8).toFixed(0)}億
DCF 公司總估算價值：${(dcfTotal/1e8).toFixed(0)} 億元
${dcfPS ? `每股 DCF 內在價值：${dcfPS.toFixed(0)} 元（目前股價 ${currentPrice} ${currentPrice < dcfPS ? '→ 低於內在價值，具安全邊際' : '→ 高於內在價值，需謹慎'}）` : '（股數資料不足，無法計算每股價值）'}`;
      }
    }

  } catch (e) { console.warn('valuation data fetch:', e.message); }

  const prompt = `請對「${stock}」進行完整估值分析。

${realData}
${peBandBlock}
${dcfBlock}

請提供：
一、多種估值法綜合比較
1. PE 法（歷史 PE Band ±1σ 區間）
2. PB 法（合理 PB 區間對應股價）
3. 殖利率回歸法（目標殖利率對應股價）
4. DCF 法（每股內在價值解讀）
5. PEG 法（本益比/成長率）

二、安全邊際與買入區間
提供「強力買入價」（-1σ以下）、「合理買入價」（均值附近）、「過熱警戒價」（+1σ以上）

三、目前估值定位
目前股價偏低/合理/偏高？與歷史均值的距離？

四、投資建議
買入 / 持有 / 賣出，目標價區間，主要風險

所屬產業：${sector}，PB：${pb}`;

  try { res.json({ result: await ask(prompt, provider, 3500) }); }
  catch (e) { res.status(500).json({ error: e.message }); }
});

/* ═══════════════════════════════════════════════════════
   ENHANCED BACKTEST: Sharpe + MaxDD + Kelly Criterion
═══════════════════════════════════════════════════════ */
app.post('/api/backtest', async (req, res) => {
  const { strategy, stock, period, capital, buyRule, sellRule, provider } = req.body;

  let metricsData = '', realMetrics = null;

  try {
    const code = stock.match(/\d{4}/)?.[0];
    const sym  = code ? code + '.TW' : /^0\d{3}/.test(stock) ? stock + '.TW' : '^TWII';
    const periodDays = { '近 1 年':365,'近 3 年':1095,'近 5 年':1825,'近 10 年':3650,'2020-2024':1825 };
    const days = periodDays[period] || 1095;
    const p1   = new Date(); p1.setDate(p1.getDate() - days);

    const hist = await yf.historical(sym, { period1: p1, interval: '1d' }, YFO);
    if (hist?.length > 30) {
      const prices  = hist.map(d => d.close).filter(Boolean);
      const returns = prices.slice(1).map((p, i) => (p - prices[i]) / prices[i]);
      const years   = days / 365;
      const total   = (prices[prices.length-1] - prices[0]) / prices[0];
      const annRet  = (Math.pow(1 + total, 1/years) - 1) * 100;

      const sharpe   = calcSharpe(returns);
      const maxDD    = calcMaxDrawdown(prices);
      const calmar   = calcCalmar(annRet, maxDD);
      const winRate  = (returns.filter(r => r > 0).length / returns.length * 100).toFixed(1);
      const meanR    = returns.reduce((a,b) => a+b, 0) / returns.length;
      const variance = returns.reduce((a,b) => a + (b-meanR)**2, 0) / returns.length;
      const annVol   = (Math.sqrt(variance) * Math.sqrt(252) * 100).toFixed(2);

      // Kelly calculation (estimated from historical win/loss)
      const wins  = returns.filter(r => r > 0);
      const loses = returns.filter(r => r < 0);
      const avgWin  = wins.length  ? wins.reduce((a,b)  => a+b,0) / wins.length  : 0;
      const avgLoss = loses.length ? Math.abs(loses.reduce((a,b) => a+b,0) / loses.length) : 0;
      const kelly   = kellyFraction(parseFloat(winRate), avgWin * 100, avgLoss * 100);
      const halfKelly = +(kelly * 0.5).toFixed(4); // Half Kelly (more conservative)

      realMetrics = {
        sharpe, maxDD, calmar, winRate, annRet: annRet.toFixed(2),
        annVol, totalRet: (total*100).toFixed(2), dataPoints: prices.length,
        kelly: +(kelly * 100).toFixed(1), halfKelly: +(halfKelly * 100).toFixed(1),
        avgWinPct: (avgWin * 100).toFixed(2), avgLossPct: (avgLoss * 100).toFixed(2),
      };

      metricsData = `
【基礎市場指標（${sym}，${period}，${hist.length} 個交易日）】
期間總報酬：${(total*100).toFixed(2)}% | 年化報酬：${annRet.toFixed(2)}%
夏普比率：${sharpe ?? '—'} | 最大回撤：${maxDD}% | 卡瑪比率：${calmar ?? '—'}
年化波動率：${annVol}% | 勝率：${winRate}%
平均盈利/虧損：${(avgWin*100).toFixed(2)}% / ${(avgLoss*100).toFixed(2)}%
凱利公式建議倉位：${(kelly*100).toFixed(1)}%（Half Kelly：${(halfKelly*100).toFixed(1)}%）`;
    }
  } catch (e) { console.warn('backtest fetch:', e.message); }

  const prompt = `請對以下策略進行深度回測分析模擬。

【策略設定】
標的：${stock} | 策略：${strategy} | 期間：${period} | 資金：${capital}
買入條件：${buyRule}
賣出條件：${sellRule}
${metricsData}

一、量化績效模擬（參考上方市場基準數據）
- 預估策略年化報酬率（vs 被動持有）
- 夏普比率預估（Sharpe < 0 不如無風險利率，0-1 尚可，1-2 良好，> 2 優秀）
- 最大回撤預估
- 交易勝率與盈虧比

二、凱利公式資金管理建議
根據策略勝率與盈虧比，解讀凱利公式建議的倉位比例是否合理？
- Full Kelly vs Half Kelly 的風險差異
- 每筆交易建議投入金額（佔總資金比例）
- 多筆同時持倉時的倉位分配

三、回測失敗案例分析（最重要）
找出3種市場環境下策略失效的情境：
情境1（市場環境 + 失敗原因 + 發生頻率）
情境2（市場環境 + 失敗原因 + 發生頻率）
情境3（市場環境 + 失敗原因 + 發生頻率）

四、AI 策略優化建議
列出3個提升夏普比率的具體改進方向

五、風險管理框架
最大單筆虧損容忍度、強制停損觸發條件、總資金最大虧損上限`;

  try {
    const result = await ask(prompt, provider, 3500);
    res.json({ result, metrics: realMetrics });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

/* ═══════════════════════════════════════════════════════
   KELLY CRITERION STANDALONE
═══════════════════════════════════════════════════════ */
app.post('/api/kelly', async (req, res) => {
  const { winRate, avgWin, avgLoss, totalCapital, maxPositions, provider } = req.body;

  const wr = parseFloat(winRate) / 100;
  const b  = Math.abs(parseFloat(avgWin) / parseFloat(avgLoss));
  const q  = 1 - wr;
  const fullKelly = (wr * b - q) / b;
  const halfKelly = fullKelly * 0.5;

  const perTrade    = Math.max(0, fullKelly) * parseFloat(totalCapital);
  const perTradeHK  = Math.max(0, halfKelly) * parseFloat(totalCapital);
  const maxSimul    = parseInt(maxPositions) || 5;
  const totalExpose = Math.max(0, halfKelly) * maxSimul * 100;

  const prompt = `請解讀以下凱利公式計算結果，並給出資金管理建議：

【輸入參數】
策略勝率：${winRate}%
平均獲利：${avgWin}%
平均虧損：${avgLoss}%
總資金：${totalCapital} 元
最大同時持倉：${maxPositions} 個

【計算結果】
Full Kelly 建議倉位：${(Math.max(0, fullKelly) * 100).toFixed(1)}%（每筆 ${perTrade.toFixed(0)} 元）
Half Kelly（推薦）：${(Math.max(0, halfKelly) * 100).toFixed(1)}%（每筆 ${perTradeHK.toFixed(0)} 元）
${maxSimul} 個同時持倉的總曝險（Half Kelly）：${totalExpose.toFixed(1)}%

請提供：
1. 這個勝率與盈虧比的整體評估（策略品質如何？）
2. 為何使用 Half Kelly 而非 Full Kelly？
3. 這個倉位比例對應的預期年化報酬率與最大回撤預估
4. 如果策略進入連虧期（5連虧），資金如何保護？
5. 不同風格的投資人（保守/穩健/積極）應如何調整這個倉位比例？
6. 實務上使用凱利公式的3個常見陷阱`;

  const calcResult = {
    fullKellyPct:  +(Math.max(0, fullKelly)  * 100).toFixed(2),
    halfKellyPct:  +(Math.max(0, halfKelly)  * 100).toFixed(2),
    perTradeFull:  +perTrade.toFixed(0),
    perTradeHalf:  +perTradeHK.toFixed(0),
    totalExposurePct: +totalExpose.toFixed(1),
    winLossRatio: +b.toFixed(3),
    expectedValue: +((wr * parseFloat(avgWin) - q * parseFloat(avgLoss))).toFixed(2),
  };

  try {
    const aiText = await ask(prompt, provider, 2000);
    res.json({ result: aiText, calc: calcResult });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

/* ═══════════════════════════════════════════════════════
   ISOLATION FOREST BLACK SWAN
═══════════════════════════════════════════════════════ */
app.post('/api/blackswan', async (req, res) => {
  const { symbols = ['2330.TW','2454.TW'], provider } = req.body;
  const ckey = 'blackswan:' + symbols.join(',');
  const cached = getCache(ckey);
  if (cached) return res.json(cached);

  try {
    const p1 = new Date(); p1.setDate(p1.getDate() - 252);
    const fetches = ['^VIX', ...symbols.slice(0, 3)].map(sym =>
      yf.historical(sym, { period1: p1, interval: '1d' }, YFO)
        .then(data => ({ sym, data })).catch(() => ({ sym, data: [] }))
    );
    const results = await Promise.all(fetches);
    const analysis = {};

    for (const { sym, data } of results) {
      if (!data || data.length < 30) continue;
      const dates   = data.map(d => d.date.toISOString().split('T')[0]);
      const closes  = data.map(d => d.close || 0);
      const volumes = data.map(d => d.volume || 0);

      const returns = closes.slice(1).map((c, i) => closes[i] ? ((c - closes[i]) / closes[i]) * 100 : 0);

      // Z-score anomalies (returns)
      const priceAnomalies = detectAnomalies(returns, dates.slice(1), 2.5, 20);
      const volAnomalies   = detectAnomalies(volumes, dates, 2.5, 20);

      // Isolation Forest on [return, volume_zscore] combined features
      if (returns.length > 50) {
        const volStats = rollingStats(volumes, 20);
        const features = returns.map((r, i) => {
          const vs = volStats[i + 1]; // offset by 1 (returns are 1 shorter)
          const vz = vs?.std && vs.std > 0 ? (volumes[i+1] - vs.mean) / vs.std : 0;
          return [r, vz];
        }).filter(f => f.length === 2);

        const ifScores = isolationForest(features, 100, Math.min(256, features.length));
        const latestScore = ifScores[ifScores.length - 1] ?? 0;
        const highIfAnomalies = ifScores.reduce((acc, s, i) => {
          if (s > 0.65) acc.push({ date: dates[i+1] || dates[dates.length-1], score: +s.toFixed(3) });
          return acc;
        }, []).slice(-5);

        const recent20Ret = returns.slice(-21);
        const latestRet   = recent20Ret[recent20Ret.length - 1];
        const m20 = recent20Ret.slice(0,-1).reduce((a,b) => a+b, 0) / 20;
        const s20 = Math.sqrt(recent20Ret.slice(0,-1).reduce((a,b) => a + (b-m20)**2, 0) / 20);
        const currentZ    = s20 > 0 ? +((latestRet - m20) / s20).toFixed(2) : 0;
        const recentV     = volumes.slice(-21);
        const latestV     = recentV[recentV.length-1];
        const vm = recentV.slice(0,-1).reduce((a,b) => a+b, 0) / 20;
        const vs = Math.sqrt(recentV.slice(0,-1).reduce((a,b) => a + (b-vm)**2, 0) / 20);
        const volZ = vs > 0 ? +((latestV - vm) / vs).toFixed(2) : 0;

        const riskLevel = latestScore > 0.7 || Math.abs(currentZ) > 3 ? 'HIGH' :
                          latestScore > 0.6 || Math.abs(currentZ) > 2 ? 'MEDIUM' : 'LOW';

        analysis[sym] = {
          currentZ, volZ,
          ifScore: +latestScore.toFixed(3),
          highIfAnomalies,
          priceAnomalies: priceAnomalies.slice(-5),
          volAnomalies: volAnomalies.slice(-5),
          totalAnomalies: priceAnomalies.length,
          riskLevel,
          latestClose: closes[closes.length-1],
          latestDate: dates[dates.length-1],
        };
      }
    }

    const summaryLines = Object.entries(analysis).map(([sym, a]) => {
      const label = sym === '^VIX' ? 'VIX 恐慌指數' : sym;
      return `${label}：Z值=${a.currentZ}，成交量Z=${a.volZ}，孤立森林分數=${a.ifScore}（>0.65為異常），風險=${a.riskLevel}，年內異常次數=${a.totalAnomalies}`;
    }).join('\n');

    const aiPrompt = `請根據以下孤立森林（Isolation Forest）與滾動Z值的市場異常偵測結果，進行黑天鵝風險評估：

【演算法結果】
${summaryLines}

孤立森林分數說明：
- < 0.5：正常（樹越難孤立它，代表越正常）
- 0.5-0.65：輕微異常
- 0.65-0.8：異常
- > 0.8：極端異常

Z 值說明：
- |Z| < 2.5：正常範圍
- |Z| 2.5-3.0：統計異常
- |Z| > 3.0：極端異常（3σ事件）

請提供：
一、整體風險評級（低/中/高/極高）與判斷邏輯
二、VIX 恐慌解讀與歷史比較
三、孤立森林異常分數解讀（哪些標的最危險？）
四、歷史類比（此類信號後市場常見走勢）
五、具體操作建議：
   - 是否應降低倉位？降低多少？
   - 是否增加避險部位（反向ETF/黃金/現金）？
   - 停損設置建議
六、Emergency Stop 觸發條件（什麼情況下應立即全數出場？）`;

    const aiAnalysis = await ask(aiPrompt, provider, 3000);
    const finalResult = { analysis, aiAnalysis, fetched: new Date().toISOString() };
    setCache(ckey, finalResult, 300_000);
    res.json(finalResult);
  } catch (e) {
    console.error('blackswan:', e.message);
    res.status(500).json({ error: e.message });
  }
});

/* ═══════════════════════════════════════════════════════
   EXISTING ENDPOINTS
═══════════════════════════════════════════════════════ */

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
        enriched = `${content}\n【即時】股價：${q.regularMarketPrice} 漲跌：${q.regularMarketChangePercent?.toFixed(2)}% PE：${q.trailingPE?.toFixed(1)||'—'} 52週高低：${q.fiftyTwoWeekHigh}/${q.fiftyTwoWeekLow}`;
      }
    } catch {}
  }
  const prompts = {
    stock:     `請根據即時資料全面分析：\n${enriched}\n\n1.公司簡介\n2.基本面（PE/殖利率/ROE）\n3.技術面走勢\n4.風險\n5.投資建議與目標價`,
    news:      `分析財經新聞投資影響：\n${content}\n\n1.摘要\n2.受影響類股\n3.短期影響\n4.長期影響\n5.應對策略`,
    portfolio: `評估投資組合：\n${content}\n\n1.多元性\n2.集中度風險\n3.預期報酬\n4.風險\n5.優化建議`,
    market:    `分析今日台股（${new Date().toLocaleDateString('zh-TW')}）：\n1.大盤走勢\n2.強勢族群\n3.弱勢族群\n4.籌碼動向\n5.明日操作重點\n6.風險提示`,
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
      histCtx = '\n【近5日歷史】\n' + hist.slice(-5).map(h =>
        `${h.date.toISOString().split('T')[0]} 收${h.close?.toFixed(0)} 量${h.volume?(h.volume/1e6).toFixed(1)+'M':'—'}`
      ).join('\n');
    }
  } catch {}
  const prompt = `請對「${stock}」進行技術指標分析。\n\n股價：${price} | MA5/20/60：${ma5}/${ma20}/${ma60}\nKD K/D：${kd_k}/${kd_d} | RSI：${rsi}\n成交量：${volume} | 走勢：${trend}${histCtx}\n\n1.均線多空排列\n2.KD指標解讀\n3.RSI強弱評估\n4.支撐與壓力位\n5.短中長期趨勢\n6.操作建議（進場/停損/目標價）`;
  try { res.json({ result: await ask(prompt, provider, 2500) }); }
  catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/allocation', async (req, res) => {
  const { holdings, risk, goal, horizon, provider } = req.body;
  const prompt = `分析並優化投資組合：\n${holdings}\n\n風險：${risk} | 目標：${goal} | 期間：${horizon}\n\n1.健診（集中度/多元性/風險評分）\n2.產業/地區配置\n3.建議最佳化配置比例\n4.減碼/增碼說明\n5.推薦新標的\n6.預期改善`;
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
  const prompt = `請生成「${month}台股市場重點摘要」：\n\n## 大盤表現\n## 產業亮點（前3強勢/前3弱勢）\n## 重要財經事件\n## 籌碼動向\n## 本月最強個股\n## 下月關鍵觀察指標\n## AI 投資建議重點\n## 風險警示\n\n請生成完整有深度的月報。`;
  try { res.json({ result: await ask(prompt, provider, 3000) }); }
  catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));
// ── Startup: warm up Yahoo Finance crumb before taking requests ──
async function warmupYahooFinance() {
  try {
    console.log('Warming up Yahoo Finance...');
    // Test direct HTTP API first (primary method)
    const test = await fetchYFPrice('AAPL');
    if (test.price > 0) {
      console.log(`Yahoo Finance HTTP ready. AAPL: ${test.price}`);
    } else {
      // Fallback: try yf library crumb init
      await yf.quote('AAPL', {}, YFO);
      console.log('Yahoo Finance library ready.');
    }
  } catch (e) {
    console.warn('Yahoo Finance warm-up failed:', e.message);
  }
}

async function populateCaches() {
  console.log('Populating ticker + watchlist caches...');

  // Fetch all ticker symbols
  const tickerResults = await Promise.all(
    TICKER_LIST.map(async (t) => {
      const q = await fetchYFPrice(t.sym);
      return { sym: t.label, ticker: t.sym, price: q.price, change: q.change, up: q.up };
    })
  );
  if (tickerResults.some(r => r.price > 0)) {
    setCache('ticker', tickerResults, 60_000);
    console.log('ticker cached:', tickerResults.map(r => `${r.sym}:${r.price}`).join(', '));
  } else {
    console.warn('ticker cache: all prices still 0');
  }

  // Fetch watchlist
  const watchlistResults = await Promise.all(WATCHLIST.map(async (w) => {
    const q = await fetchYFPrice(w.sym);
    const price  = q.price  > 0 ? q.price  : null;
    const change = q.price  > 0 ? q.change : null;
    let pe = null, mktCap = null, week52H = null, week52L = null;
    try {
      const detail = await yf.quote(w.sym, {}, YFO);
      pe = detail.trailingPE ?? null;
      mktCap = detail.marketCap ?? null;
      week52H = detail.fiftyTwoWeekHigh ?? null;
      week52L = detail.fiftyTwoWeekLow  ?? null;
    } catch {}
    return { ...w, price, change, up: change != null ? change >= 0 : true, pe, mktCap, week52H, week52L };
  }));
  if (watchlistResults.some(r => r.price != null && r.price > 0)) {
    setCache('watchlist', watchlistResults, 60_000);
    console.log('watchlist cached:', watchlistResults.map(r => `${r.code}:${r.price}`).join(', '));
  } else {
    console.warn('watchlist cache: all prices still 0');
  }
}

app.listen(PORT, async () => {
  console.log(`🚀 智投 AI — http://localhost:${PORT}`);
  await warmupYahooFinance();
  await populateCaches();
  // Refresh caches every 60s
  setInterval(populateCaches, 60_000);
});