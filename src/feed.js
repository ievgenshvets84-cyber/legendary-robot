const SHOP_DOMAIN = process.env.SHOP_DOMAIN || 'https://www.estid-fashion.de';
const CURRENCY = process.env.FEED_CURRENCY || 'EUR';
const LANGUAGE = process.env.FEED_LANGUAGE || 'de';
const COUNTRY = process.env.FEED_COUNTRY || 'DE';

// Shopify-Produkttyp → Google Shopping-Kategorie
const CATEGORY_MAP = {
  Earrings: 'Bekleidung & Accessoires > Schmuck > Ohrringe',
  Necklaces: 'Bekleidung & Accessoires > Schmuck > Halsketten',
  Rings: 'Bekleidung & Accessoires > Schmuck > Ringe',
  Bracelets: 'Bekleidung & Accessoires > Schmuck > Armbänder',
  Brooches: 'Bekleidung & Accessoires > Schmuck > Broschen',
  Jewelry: 'Bekleidung & Accessoires > Schmuck',
};

function getGoogleCategory(productType) {
  return CATEGORY_MAP[productType] || 'Bekleidung & Accessoires > Schmuck';
}

function stripHtml(html) {
  return (html || '').replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
}

function buildProductUrl(handle) {
  return `${SHOP_DOMAIN}/products/${handle}`;
}

function formatPrice(amount, currency = CURRENCY) {
  return `${parseFloat(amount).toFixed(2)} ${currency}`;
}

function escapeXml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function buildItemXml(product, variant) {
  const itemId = variant.sku || `shopify-${product.id.split('/').pop()}-${variant.id.split('/').pop()}`;
  const title = product.variants?.length > 1
    ? `${product.title} – ${variant.title}`
    : product.title;

  const description = stripHtml(product.descriptionHtml).substring(0, 5000);
  const link = buildProductUrl(product.handle);
  const imageUrl = product.featuredImage?.url || '';
  const price = formatPrice(variant.price);
  const compareAtPrice = variant.compareAtPrice
    ? formatPrice(variant.compareAtPrice)
    : null;
  const availability = variant.availableForSale ? 'in stock' : 'out of stock';
  const condition = 'new';
  const brand = escapeXml(product.vendor || 'ESTID Fashion');
  const gtin = variant.barcode || '';
  const googleCategory = getGoogleCategory(product.productType);

  // Zusatzbilder
  const additionalImages = (product.images?.edges || [])
    .map(e => e.node.url)
    .filter(url => url !== imageUrl)
    .slice(0, 9);

  const additionalImageTags = additionalImages
    .map(url => `    <g:additional_image_link>${escapeXml(url)}</g:additional_image_link>`)
    .join('\n');

  const salePriceTag = compareAtPrice
    ? `    <g:sale_price>${escapeXml(price)}</g:sale_price>\n    <g:price>${escapeXml(compareAtPrice)}</g:price>`
    : `    <g:price>${escapeXml(price)}</g:price>`;

  // Varianten-Attribute
  const variantAttrs = (variant.selectedOptions || [])
    .map(opt => {
      const name = opt.name.toLowerCase();
      if (name === 'size' || name === 'größe') return `    <g:size>${escapeXml(opt.value)}</g:size>`;
      if (name === 'color' || name === 'farbe') return `    <g:color>${escapeXml(opt.value)}</g:color>`;
      return '';
    })
    .filter(Boolean)
    .join('\n');

  return `  <item>
    <g:id>${escapeXml(itemId)}</g:id>
    <g:title>${escapeXml(title)}</g:title>
    <g:description>${escapeXml(description)}</g:description>
    <g:link>${escapeXml(link)}</g:link>
    <g:image_link>${escapeXml(imageUrl)}</g:image_link>
${additionalImageTags ? additionalImageTags + '\n' : ''}    <g:availability>${availability}</g:availability>
    <g:condition>${condition}</g:condition>
${salePriceTag}
    <g:brand>${brand}</g:brand>
    <g:google_product_category>${escapeXml(googleCategory)}</g:google_product_category>
    <g:product_type>${escapeXml(product.productType || 'Schmuck')}</g:product_type>
${gtin ? `    <g:gtin>${escapeXml(gtin)}</g:gtin>\n` : ''}${variant.sku ? `    <g:mpn>${escapeXml(variant.sku)}</g:mpn>\n` : ''}    <g:identifier_exists>${gtin ? 'yes' : 'no'}</g:identifier_exists>
    <g:shipping>
      <g:country>${COUNTRY}</g:country>
      <g:service>Standardversand</g:service>
      <g:price>4.95 ${CURRENCY}</g:price>
    </g:shipping>
    <g:shipping_weight>${variant.weight ? `${variant.weight} ${(variant.weightUnit || 'g').toLowerCase()}` : '0.1 kg'}</g:shipping_weight>
${variantAttrs ? variantAttrs + '\n' : ''}  </item>`;
}

function buildFeedXml(products) {
  const now = new Date().toISOString();
  const items = [];

  for (const product of products) {
    const variants = product.variants?.edges?.map(e => e.node) || [];
    for (const variant of variants) {
      items.push(buildItemXml(product, variant));
    }
  }

  return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:g="http://base.google.com/ns/1.0">
  <channel>
    <title>ESTID Fashion – Google Shopping Feed</title>
    <link>${SHOP_DOMAIN}</link>
    <description>Schmuck &amp; Mode von ESTID Fashion – ${SHOP_DOMAIN}</description>
    <language>${LANGUAGE}-${COUNTRY.toLowerCase()}</language>
    <lastBuildDate>${now}</lastBuildDate>
${items.join('\n')}
  </channel>
</rss>`;
}

module.exports = { buildFeedXml };
