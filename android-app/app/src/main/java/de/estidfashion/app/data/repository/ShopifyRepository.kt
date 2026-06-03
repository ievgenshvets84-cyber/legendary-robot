package de.estidfashion.app.data.repository

import de.estidfashion.app.data.api.GraphQLRequest
import de.estidfashion.app.data.api.ShopifyQueries
import de.estidfashion.app.data.api.ShopifyStorefrontApi
import de.estidfashion.app.data.models.*
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ShopifyRepository @Inject constructor(
    private val api: ShopifyStorefrontApi,
) {

    suspend fun getProducts(
        query: String? = null,
        cursor: String? = null,
        pageSize: Int = 20,
    ): Result<Pair<List<Product>, String?>> = runCatching {
        val res = api.graphql(
            GraphQLRequest(
                query = ShopifyQueries.PRODUCTS,
                variables = mapOf("first" to pageSize, "after" to cursor, "query" to query),
            )
        )
        val productsData = res.data?.get("products") as? Map<*, *>
        val edges = productsData?.get("edges") as? List<*> ?: emptyList<Any>()
        val pageInfo = productsData?.get("pageInfo") as? Map<*, *>
        val nextCursor = if (pageInfo?.get("hasNextPage") == true)
            pageInfo["endCursor"] as? String else null

        val products = edges.mapNotNull { edge ->
            (edge as? Map<*, *>)?.get("node")?.let { parseProduct(it as Map<*, *>) }
        }
        Pair(products, nextCursor)
    }

    suspend fun getProductByHandle(handle: String): Result<Product> = runCatching {
        val res = api.graphql(
            GraphQLRequest(
                query = ShopifyQueries.PRODUCT_BY_HANDLE,
                variables = mapOf("handle" to handle),
            )
        )
        val node = res.data?.get("productByHandle") as? Map<*, *>
            ?: throw Exception("Produkt nicht gefunden")
        parseProduct(node)
    }

    suspend fun getCollections(): Result<List<Collection>> = runCatching {
        val res = api.graphql(
            GraphQLRequest(
                query = ShopifyQueries.COLLECTIONS,
                variables = mapOf("first" to 20),
            )
        )
        val edges = (res.data?.get("collections") as? Map<*, *>)
            ?.get("edges") as? List<*> ?: emptyList<Any>()
        edges.mapNotNull { edge ->
            (edge as? Map<*, *>)?.get("node")?.let { n ->
                val node = n as Map<*, *>
                Collection(
                    id = node["id"] as? String ?: "",
                    title = node["title"] as? String ?: "",
                    handle = node["handle"] as? String ?: "",
                    description = node["description"] as? String ?: "",
                    imageUrl = (node["image"] as? Map<*, *>)?.get("url") as? String,
                    productsCount = ((node["products"] as? Map<*, *>)?.get("totalCount") as? Double)?.toInt() ?: 0,
                )
            }
        }
    }

    suspend fun createCheckout(cartItems: List<CartItem>): Result<String> = runCatching {
        val lines = cartItems.map { mapOf("merchandiseId" to it.variantId, "quantity" to it.quantity) }
        val res = api.graphql(
            GraphQLRequest(
                query = ShopifyQueries.CREATE_CART,
                variables = mapOf("lines" to lines),
            )
        )
        val cart = (res.data?.get("cartCreate") as? Map<*, *>)?.get("cart") as? Map<*, *>
        cart?.get("checkoutUrl") as? String ?: throw Exception("Checkout konnte nicht erstellt werden")
    }

    suspend fun searchProducts(query: String): Result<List<Product>> = runCatching {
        val res = api.graphql(
            GraphQLRequest(
                query = ShopifyQueries.SEARCH_PRODUCTS,
                variables = mapOf("query" to "title:*${query}* OR tag:${query}", "first" to 20),
            )
        )
        val edges = (res.data?.get("products") as? Map<*, *>)
            ?.get("edges") as? List<*> ?: emptyList<Any>()
        edges.mapNotNull { edge ->
            (edge as? Map<*, *>)?.get("node")?.let { parseProduct(it as Map<*, *>) }
        }
    }

    // ── Parser ────────────────────────────────────────────────────────────────

    @Suppress("UNCHECKED_CAST")
    private fun parseProduct(node: Map<*, *>): Product {
        val priceRange = node["priceRange"] as? Map<*, *>
        val minPrice = ((priceRange?.get("minVariantPrice") as? Map<*, *>)?.get("amount") as? String)?.toDoubleOrNull() ?: 0.0
        val maxPrice = ((priceRange?.get("maxVariantPrice") as? Map<*, *>)?.get("amount") as? String)?.toDoubleOrNull() ?: minPrice
        val currency = ((priceRange?.get("minVariantPrice") as? Map<*, *>)?.get("currencyCode") as? String) ?: "EUR"

        val images = (node["images"] as? Map<*, *>)
            ?.get("edges")?.let { it as? List<*> }
            ?.mapNotNull { ((it as? Map<*, *>)?.get("node") as? Map<*, *>)?.get("url") as? String }
            ?: emptyList()

        val variants = (node["variants"] as? Map<*, *>)
            ?.get("edges")?.let { it as? List<*> }
            ?.mapNotNull { edge ->
                (edge as? Map<*, *>)?.get("node")?.let { v ->
                    val vn = v as Map<*, *>
                    ProductVariant(
                        id = vn["id"] as? String ?: "",
                        title = vn["title"] as? String ?: "",
                        sku = vn["sku"] as? String ?: "",
                        price = ((vn["price"] as? Map<*, *>)?.get("amount") as? String)?.toDoubleOrNull() ?: 0.0,
                        compareAtPrice = ((vn["compareAtPrice"] as? Map<*, *>)?.get("amount") as? String)?.toDoubleOrNull(),
                        availableForSale = vn["availableForSale"] as? Boolean ?: false,
                        quantityAvailable = (vn["quantityAvailable"] as? Double)?.toInt() ?: 0,
                        selectedOptions = (vn["selectedOptions"] as? List<*>)
                            ?.mapNotNull { opt ->
                                (opt as? Map<*, *>)?.let {
                                    SelectedOption(it["name"] as? String ?: "", it["value"] as? String ?: "")
                                }
                            } ?: emptyList(),
                    )
                }
            } ?: emptyList()

        return Product(
            id = node["id"] as? String ?: "",
            title = node["title"] as? String ?: "",
            handle = node["handle"] as? String ?: "",
            description = (node["description"] ?: node["descriptionHtml"]) as? String ?: "",
            vendor = node["vendor"] as? String ?: "ESTID Fashion",
            productType = node["productType"] as? String ?: "",
            tags = (node["tags"] as? List<*>)?.filterIsInstance<String>() ?: emptyList(),
            featuredImageUrl = (node["featuredImage"] as? Map<*, *>)?.get("url") as? String
                ?: images.firstOrNull(),
            images = images,
            variants = variants,
            priceMin = minPrice,
            priceMax = maxPrice,
            currency = currency,
            availableForSale = node["availableForSale"] as? Boolean ?: true,
        )
    }
}
