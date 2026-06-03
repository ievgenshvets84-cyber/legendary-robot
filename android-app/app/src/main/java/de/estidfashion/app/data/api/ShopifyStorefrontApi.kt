package de.estidfashion.app.data.api

import retrofit2.http.Body
import retrofit2.http.POST

// Shopify Storefront API – GraphQL über Retrofit
interface ShopifyStorefrontApi {

    @POST("api/2024-01/graphql.json")
    suspend fun graphql(@Body request: GraphQLRequest): GraphQLResponse
}

data class GraphQLRequest(
    val query: String,
    val variables: Map<String, Any?> = emptyMap(),
)

data class GraphQLResponse(
    val data: Map<String, Any?>?,
    val errors: List<Map<String, Any?>>?,
)

// ── GraphQL Queries ──────────────────────────────────────────────────────────

object ShopifyQueries {

    val SHOP_INFO = """
        query {
          shop {
            name
            primaryDomain { url }
            description
            paymentSettings { currencyCode }
          }
        }
    """.trimIndent()

    val COLLECTIONS = """
        query GetCollections(${"$"}first: Int!) {
          collections(first: ${"$"}first) {
            edges {
              node {
                id title handle description
                image { url }
                products { totalCount }
              }
            }
          }
        }
    """.trimIndent()

    val PRODUCTS = """
        query GetProducts(${"$"}first: Int!, ${"$"}after: String, ${"$"}query: String) {
          products(first: ${"$"}first, after: ${"$"}after, query: ${"$"}query) {
            pageInfo { hasNextPage endCursor }
            edges {
              node {
                id title handle
                description(truncateAt: 300)
                vendor productType
                tags
                availableForSale
                featuredImage { url }
                images(first: 5) { edges { node { url } } }
                priceRange {
                  minVariantPrice { amount currencyCode }
                  maxVariantPrice { amount currencyCode }
                }
                variants(first: 10) {
                  edges {
                    node {
                      id title sku availableForSale
                      quantityAvailable
                      price { amount }
                      compareAtPrice { amount }
                      selectedOptions { name value }
                    }
                  }
                }
              }
            }
          }
        }
    """.trimIndent()

    val PRODUCT_BY_HANDLE = """
        query GetProduct(${"$"}handle: String!) {
          productByHandle(handle: ${"$"}handle) {
            id title handle
            descriptionHtml
            vendor productType tags
            availableForSale
            images(first: 10) { edges { node { url altText } } }
            priceRange {
              minVariantPrice { amount currencyCode }
              maxVariantPrice { amount currencyCode }
            }
            variants(first: 20) {
              edges {
                node {
                  id title sku availableForSale
                  quantityAvailable
                  price { amount }
                  compareAtPrice { amount }
                  selectedOptions { name value }
                }
              }
            }
          }
        }
    """.trimIndent()

    val CREATE_CART = """
        mutation CreateCart(${"$"}lines: [CartLineInput!]) {
          cartCreate(input: { lines: ${"$"}lines }) {
            cart {
              id checkoutUrl
              cost { totalAmount { amount currencyCode } }
            }
            userErrors { field message }
          }
        }
    """.trimIndent()

    val SEARCH_PRODUCTS = """
        query SearchProducts(${"$"}query: String!, ${"$"}first: Int!) {
          products(first: ${"$"}first, query: ${"$"}query) {
            edges {
              node {
                id title handle
                description(truncateAt: 200)
                featuredImage { url }
                priceRange {
                  minVariantPrice { amount currencyCode }
                }
                availableForSale
                variants(first: 1) {
                  edges { node { id price { amount } availableForSale } }
                }
              }
            }
          }
        }
    """.trimIndent()
}
