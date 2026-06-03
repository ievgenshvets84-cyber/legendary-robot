const fetch = require('node-fetch');

const MERCHANT_ID = process.env.GOOGLE_MERCHANT_ID;
const BASE_URL = 'https://shoppingcontent.googleapis.com/content/v2.1';

async function getAccessToken() {
  const res = await fetch('https://oauth2.googleapis.com/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      client_id: process.env.GOOGLE_CLIENT_ID,
      client_secret: process.env.GOOGLE_CLIENT_SECRET,
      refresh_token: process.env.GOOGLE_REFRESH_TOKEN,
      grant_type: 'refresh_token',
    }),
  });
  const data = await res.json();
  if (!data.access_token) throw new Error('Google OAuth fehlgeschlagen: ' + JSON.stringify(data));
  return data.access_token;
}

async function merchantRequest(method, path, body = null) {
  const token = await getAccessToken();
  const res = await fetch(`${BASE_URL}/${MERCHANT_ID}${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const json = await res.json();
  if (!res.ok) throw new Error(`Merchant API Fehler: ${JSON.stringify(json)}`);
  return json;
}

function shopifyProductToMerchantItem(product, variant) {
  const SHOP_DOMAIN = process.env.SHOP_DOMAIN || 'https://www.estid-fashion.de';
  const id = variant.sku || `shopify-${product.id.split('/').pop()}`;

  return {
    kind: 'content#product',
    offerId: id,
    title: product.title,
    description: (product.descriptionHtml || '').replace(/<[^>]*>/g, ' ').substring(0, 5000),
    link: `${SHOP_DOMAIN}/products/${product.handle}`,
    imageLink: product.featuredImage?.url,
    contentLanguage: 'de',
    targetCountry: 'DE',
    channel: 'online',
    availability: variant.availableForSale ? 'in stock' : 'out of stock',
    condition: 'new',
    price: { value: parseFloat(variant.price).toFixed(2), currency: 'EUR' },
    brand: product.vendor || 'ESTID Fashion',
    mpn: variant.sku,
    googleProductCategory: 'Bekleidung & Accessoires > Schmuck',
    productType: product.productType || 'Schmuck',
    shipping: [{ country: 'DE', service: 'Standardversand', price: { value: '4.95', currency: 'EUR' } }],
    identifierExists: false,
  };
}

async function insertProduct(product, variant) {
  const item = shopifyProductToMerchantItem(product, variant);
  return merchantRequest('POST', '/products', item);
}

async function deleteProduct(offerId) {
  return merchantRequest('DELETE', `/products/online:de:DE:${offerId}`);
}

async function listMerchantProducts(maxResults = 50) {
  return merchantRequest('GET', `/products?maxResults=${maxResults}`);
}

async function syncAllProducts(products) {
  const results = { success: 0, failed: 0, errors: [] };

  // Batch-Insert in Gruppen von 1000
  const entries = [];
  for (const product of products) {
    const variants = product.variants?.edges?.map(e => e.node) || [];
    for (const variant of variants) {
      entries.push({
        batchId: entries.length + 1,
        merchantId: parseInt(MERCHANT_ID),
        method: 'insert',
        product: shopifyProductToMerchantItem(product, variant),
      });
    }
  }

  // In Pakete à 1000 aufteilen
  for (let i = 0; i < entries.length; i += 1000) {
    const batch = entries.slice(i, i + 1000);
    try {
      const res = await merchantRequest('POST', '/../products/batch', { entries: batch });
      for (const entry of res.entries || []) {
        if (entry.errors) {
          results.failed++;
          results.errors.push({ id: entry.batchId, errors: entry.errors.errors });
        } else {
          results.success++;
        }
      }
    } catch (err) {
      results.failed += batch.length;
      results.errors.push({ batch: i, error: err.message });
    }
  }

  return results;
}

module.exports = { insertProduct, deleteProduct, listMerchantProducts, syncAllProducts };
