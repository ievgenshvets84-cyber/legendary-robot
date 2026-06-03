require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const cron = require('node-cron');
const { getAllProducts, getShopInfo } = require('./src/shopify');
const { buildFeedXml } = require('./src/feed');
const merchant = require('./src/google-merchant');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Cache für Feed
let feedCache = null;
let feedCachedAt = null;
const CACHE_TTL_MS = 30 * 60 * 1000; // 30 Minuten

async function getFeedCached() {
  const now = Date.now();
  if (feedCache && feedCachedAt && now - feedCachedAt < CACHE_TTL_MS) {
    return feedCache;
  }
  const products = await getAllProducts();
  feedCache = buildFeedXml(products);
  feedCachedAt = now;
  return feedCache;
}

// ── Google Shopping XML-Feed ──────────────────────────────────────────────────
// Diese URL gibst du bei Google Merchant Center als Feed-URL an:
// https://deine-app.com/feed/google-shopping.xml
app.get('/feed/google-shopping.xml', async (req, res) => {
  try {
    const xml = await getFeedCached();
    res.set('Content-Type', 'application/xml; charset=utf-8');
    res.set('Cache-Control', 'public, max-age=1800');
    res.send(xml);
  } catch (err) {
    console.error('[Feed] Fehler:', err.message);
    res.status(500).send('Feed konnte nicht generiert werden: ' + err.message);
  }
});

// ── API: Shop-Info ────────────────────────────────────────────────────────────
app.get('/api/shop', async (req, res) => {
  try {
    const info = await getShopInfo();
    res.json(info);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ── API: Produkte ─────────────────────────────────────────────────────────────
app.get('/api/products', async (req, res) => {
  try {
    const products = await getAllProducts();
    res.json({ count: products.length, products });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ── API: Feed-Vorschau ────────────────────────────────────────────────────────
app.get('/api/feed-preview', async (req, res) => {
  try {
    feedCache = null; // Cache leeren für frischen Feed
    const xml = await getFeedCached();
    res.json({ xml: xml.substring(0, 5000) + '...', generatedAt: new Date(feedCachedAt).toISOString() });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ── API: Google Merchant Center Sync ─────────────────────────────────────────
app.post('/api/merchant/sync', async (req, res) => {
  if (!process.env.GOOGLE_MERCHANT_ID) {
    return res.status(400).json({ error: 'GOOGLE_MERCHANT_ID nicht konfiguriert' });
  }
  try {
    const products = await getAllProducts();
    const result = await merchant.syncAllProducts(products);
    res.json({ message: 'Sync abgeschlossen', ...result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ── API: Google Merchant Produkte anzeigen ────────────────────────────────────
app.get('/api/merchant/products', async (req, res) => {
  if (!process.env.GOOGLE_MERCHANT_ID) {
    return res.status(400).json({ error: 'GOOGLE_MERCHANT_ID nicht konfiguriert' });
  }
  try {
    const data = await merchant.listMerchantProducts();
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ── API: Feed-Cache leeren ────────────────────────────────────────────────────
app.post('/api/cache/clear', (req, res) => {
  feedCache = null;
  feedCachedAt = null;
  res.json({ message: 'Cache geleert' });
});

// ── API: Status ───────────────────────────────────────────────────────────────
app.get('/api/status', (req, res) => {
  res.json({
    status: 'ok',
    shop: process.env.SHOPIFY_STORE_URL || 'nicht konfiguriert',
    merchantId: process.env.GOOGLE_MERCHANT_ID || 'nicht konfiguriert',
    feedUrl: `${process.env.APP_URL || 'http://localhost:' + PORT}/feed/google-shopping.xml`,
    cacheAlter: feedCachedAt ? Math.round((Date.now() - feedCachedAt) / 1000) + 's' : 'kein Cache',
    shopDomain: process.env.SHOP_DOMAIN || 'https://www.estid-fashion.de',
  });
});

// ── Dashboard ─────────────────────────────────────────────────────────────────
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ── Automatischer Feed-Refresh alle 30 Minuten ────────────────────────────────
cron.schedule('*/30 * * * *', async () => {
  console.log('[Cron] Feed wird aktualisiert...');
  feedCache = null;
  try {
    await getFeedCached();
    console.log('[Cron] Feed erfolgreich aktualisiert');
  } catch (err) {
    console.error('[Cron] Fehler beim Feed-Update:', err.message);
  }
});

app.listen(PORT, () => {
  console.log(`\n🛍️  ESTID Fashion – Google App läuft auf Port ${PORT}`);
  console.log(`📊  Dashboard:  http://localhost:${PORT}`);
  console.log(`📦  Feed URL:   http://localhost:${PORT}/feed/google-shopping.xml`);
  console.log(`🔗  Shop:       https://www.estid-fashion.de\n`);
});
