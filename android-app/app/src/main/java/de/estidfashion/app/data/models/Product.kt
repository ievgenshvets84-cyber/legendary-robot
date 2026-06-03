package de.estidfashion.app.data.models

import androidx.room.Entity
import androidx.room.PrimaryKey

data class Product(
    val id: String,
    val title: String,
    val handle: String,
    val description: String,
    val vendor: String,
    val productType: String,
    val tags: List<String>,
    val featuredImageUrl: String?,
    val images: List<String>,
    val variants: List<ProductVariant>,
    val priceMin: Double,
    val priceMax: Double,
    val currency: String,
    val availableForSale: Boolean,
) {
    val primaryVariant get() = variants.firstOrNull()
    val displayPrice get() = if (priceMin == priceMax)
        "%.2f €".format(priceMin)
    else
        "%.2f – %.2f €".format(priceMin, priceMax)
    val isOnSale get() = variants.any { it.compareAtPrice != null && it.compareAtPrice > it.price }
}

data class ProductVariant(
    val id: String,
    val title: String,
    val sku: String,
    val price: Double,
    val compareAtPrice: Double?,
    val availableForSale: Boolean,
    val quantityAvailable: Int,
    val selectedOptions: List<SelectedOption>,
) {
    val discountPercent: Int? get() = compareAtPrice?.let {
        if (it > price) ((1.0 - price / it) * 100).toInt() else null
    }
}

data class SelectedOption(val name: String, val value: String)

data class Collection(
    val id: String,
    val title: String,
    val handle: String,
    val description: String,
    val imageUrl: String?,
    val productsCount: Int,
)

@Entity(tableName = "cart_items")
data class CartItem(
    @PrimaryKey val variantId: String,
    val productId: String,
    val title: String,
    val variantTitle: String,
    val price: Double,
    val imageUrl: String?,
    var quantity: Int,
) {
    val subtotal get() = price * quantity
}

data class Order(
    val id: String,
    val orderNumber: String,
    val createdAt: String,
    val totalPrice: Double,
    val currency: String,
    val financialStatus: String,
    val fulfillmentStatus: String?,
    val lineItems: List<OrderLineItem>,
)

data class OrderLineItem(
    val title: String,
    val quantity: Int,
    val price: Double,
)

data class Address(
    val firstName: String = "",
    val lastName: String = "",
    val address1: String = "",
    val address2: String = "",
    val city: String = "",
    val zip: String = "",
    val country: String = "Germany",
    val countryCode: String = "DE",
    val phone: String = "",
)
