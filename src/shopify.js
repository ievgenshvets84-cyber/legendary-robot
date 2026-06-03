const fetch = require('node-fetch');

const SHOPIFY_API_VERSION = process.env.SHOPIFY_API_VERSION || '2024-01';

async function shopifyGraphQL(query, variables = {}) {
  const url = `https://${process.env.SHOPIFY_STORE_URL}/admin/api/${SHOPIFY_API_VERSION}/graphql.json`;

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Shopify-Access-Token': process.env.SHOPIFY_ACCESS_TOKEN,
    },
    body: JSON.stringify({ query, variables }),
  });

  if (!res.ok) throw new Error(`Shopify API Fehler: ${res.status} ${res.statusText}`);
  const json = await res.json();
  if (json.errors) throw new Error(`GraphQL Fehler: ${JSON.stringify(json.errors)}`);
  return json.data;
}

async function getAllProducts() {
  const products = [];
  let cursor = null;
  let hasNextPage = true;

  const PRODUCTS_QUERY = `
    query getProducts($first: Int!, $after: String) {
      products(first: $first, after: $after, query: "status:active") {
        pageInfo { hasNextPage endCursor }
        edges {
          node {
            id
            title
            handle
            descriptionHtml
            productType
            vendor
            tags
            status
            onlineStoreUrl
            createdAt
            updatedAt
            featuredImage { url altText }
            images(first: 5) {
              edges { node { url altText } }
            }
            priceRangeV2 {
              minVariantPrice { amount currencyCode }
              maxVariantPrice { amount currencyCode }
            }
            variants(first: 10) {
              edges {
                node {
                  id
                  title
                  sku
                  price
                  compareAtPrice
                  availableForSale
                  inventoryQuantity
                  barcode
                  weight
                  weightUnit
                  selectedOptions { name value }
                }
              }
            }
          }
        }
      }
    }
  `;

  while (hasNextPage) {
    const data = await shopifyGraphQL(PRODUCTS_QUERY, {
      first: 50,
      after: cursor,
    });

    const page = data.products;
    for (const edge of page.edges) {
      products.push(edge.node);
    }

    hasNextPage = page.pageInfo.hasNextPage;
    cursor = page.pageInfo.endCursor;
  }

  return products;
}

async function getShopInfo() {
  const data = await shopifyGraphQL(`
    query {
      shop {
        name
        primaryDomain { url }
        email
        currencyCode
        billingAddress { countryCodeV2 }
      }
    }
  `);
  return data.shop;
}

module.exports = { getAllProducts, getShopInfo, shopifyGraphQL };
