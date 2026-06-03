package de.estidfashion.app.viewmodel

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import de.estidfashion.app.data.models.CartItem
import de.estidfashion.app.data.repository.CartRepository
import de.estidfashion.app.data.repository.ShopifyRepository
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class CartViewModel @Inject constructor(
    private val cartRepo: CartRepository,
    private val shopifyRepo: ShopifyRepository,
) : ViewModel() {

    val cartItems: StateFlow<List<CartItem>> = cartRepo.cartItems
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    val itemCount: StateFlow<Int> = cartRepo.itemCount
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), 0)

    val total: StateFlow<Double> = cartRepo.total
        .map { it ?: 0.0 }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), 0.0)

    private val _checkoutUrl = MutableStateFlow<String?>(null)
    val checkoutUrl: StateFlow<String?> = _checkoutUrl.asStateFlow()

    private val _isCheckingOut = MutableStateFlow(false)
    val isCheckingOut: StateFlow<Boolean> = _isCheckingOut.asStateFlow()

    fun addToCart(item: CartItem) = viewModelScope.launch {
        cartRepo.addToCart(item)
    }

    fun updateQuantity(variantId: String, quantity: Int) = viewModelScope.launch {
        cartRepo.updateQuantity(variantId, quantity)
    }

    fun removeItem(variantId: String) = viewModelScope.launch {
        cartRepo.removeItem(variantId)
    }

    fun clearCart() = viewModelScope.launch { cartRepo.clearCart() }

    fun startCheckout(context: Context) {
        viewModelScope.launch {
            _isCheckingOut.value = true
            val items = cartItems.value
            if (items.isEmpty()) { _isCheckingOut.value = false; return@launch }
            shopifyRepo.createCheckout(items).fold(
                onSuccess = { url ->
                    // Shopify Checkout im Browser öffnen
                    val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
                    intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    context.startActivity(intent)
                    clearCart()
                },
                onFailure = { _checkoutUrl.value = "error" }
            )
            _isCheckingOut.value = false
        }
    }
}
