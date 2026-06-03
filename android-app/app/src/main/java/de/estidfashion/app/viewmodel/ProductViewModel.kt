package de.estidfashion.app.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dagger.hilt.android.lifecycle.HiltViewModel
import de.estidfashion.app.data.models.Collection
import de.estidfashion.app.data.models.Product
import de.estidfashion.app.data.repository.ShopifyRepository
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ProductUiState(
    val products: List<Product> = emptyList(),
    val isLoading: Boolean = false,
    val isLoadingMore: Boolean = false,
    val error: String? = null,
    val hasNextPage: Boolean = false,
    val nextCursor: String? = null,
    val collections: List<Collection> = emptyList(),
    val selectedCollection: String? = null,
    val searchQuery: String = "",
    val filterType: String? = null,
)

@HiltViewModel
class ProductViewModel @Inject constructor(
    private val repo: ShopifyRepository,
) : ViewModel() {

    private val _state = MutableStateFlow(ProductUiState(isLoading = true))
    val state: StateFlow<ProductUiState> = _state.asStateFlow()

    init {
        loadCollections()
        loadProducts()
    }

    fun loadProducts(reset: Boolean = true) {
        viewModelScope.launch {
            if (reset) _state.update { it.copy(isLoading = true, error = null, products = emptyList()) }
            else _state.update { it.copy(isLoadingMore = true) }

            val query = buildQuery()
            val cursor = if (reset) null else _state.value.nextCursor
            repo.getProducts(query = query, cursor = cursor).fold(
                onSuccess = { (products, nextCursor) ->
                    _state.update { s ->
                        s.copy(
                            products = if (reset) products else s.products + products,
                            isLoading = false,
                            isLoadingMore = false,
                            hasNextPage = nextCursor != null,
                            nextCursor = nextCursor,
                        )
                    }
                },
                onFailure = { err ->
                    _state.update { it.copy(isLoading = false, isLoadingMore = false, error = err.message) }
                }
            )
        }
    }

    fun loadMore() {
        if (_state.value.hasNextPage && !_state.value.isLoadingMore) loadProducts(reset = false)
    }

    private fun loadCollections() {
        viewModelScope.launch {
            repo.getCollections().onSuccess { cols ->
                _state.update { it.copy(collections = cols) }
            }
        }
    }

    fun search(query: String) {
        _state.update { it.copy(searchQuery = query) }
        if (query.length >= 2 || query.isEmpty()) loadProducts()
    }

    fun selectCollection(handle: String?) {
        _state.update { it.copy(selectedCollection = handle, searchQuery = "") }
        loadProducts()
    }

    fun selectFilter(type: String?) {
        _state.update { it.copy(filterType = type) }
        loadProducts()
    }

    private fun buildQuery(): String? {
        val parts = mutableListOf<String>()
        parts += "status:active"
        _state.value.selectedCollection?.let { parts += "collection:$it" }
        _state.value.filterType?.let { parts += "product_type:$it" }
        _state.value.searchQuery.takeIf { it.isNotBlank() }?.let { parts += "title:*$it*" }
        return parts.joinToString(" AND ")
    }

    fun clearError() = _state.update { it.copy(error = null) }
}

@HiltViewModel
class ProductDetailViewModel @Inject constructor(
    private val repo: ShopifyRepository,
) : ViewModel() {

    private val _product = MutableStateFlow<Product?>(null)
    val product: StateFlow<Product?> = _product.asStateFlow()

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    private val _selectedVariantIndex = MutableStateFlow(0)
    val selectedVariantIndex: StateFlow<Int> = _selectedVariantIndex.asStateFlow()

    fun load(handle: String) {
        viewModelScope.launch {
            _isLoading.value = true
            repo.getProductByHandle(handle).onSuccess { p ->
                _product.value = p
                _isLoading.value = false
            }.onFailure {
                _isLoading.value = false
            }
        }
    }

    fun selectVariant(index: Int) { _selectedVariantIndex.value = index }

    val selectedVariant get() = product.value?.let { p ->
        p.variants.getOrNull(_selectedVariantIndex.value) ?: p.variants.firstOrNull()
    }
}
