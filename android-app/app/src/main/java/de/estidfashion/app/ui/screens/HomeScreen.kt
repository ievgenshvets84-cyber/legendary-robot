package de.estidfashion.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import coil.compose.AsyncImage
import de.estidfashion.app.ui.components.ProductCard
import de.estidfashion.app.ui.theme.EstidGold
import de.estidfashion.app.viewmodel.CartViewModel
import de.estidfashion.app.viewmodel.ProductViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    onProductClick: (String) -> Unit,
    onCartClick: () -> Unit,
    productVm: ProductViewModel = hiltViewModel(),
    cartVm: CartViewModel = hiltViewModel(),
) {
    val state by productVm.state.collectAsStateWithLifecycle()
    val cartCount by cartVm.itemCount.collectAsStateWithLifecycle()
    var searchText by remember { mutableStateOf("") }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            "ESTID Fashion",
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                        )
                        Text(
                            "estid-fashion.de",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                },
                actions = {
                    IconButton(onClick = onCartClick) {
                        BadgedBox(badge = {
                            if (cartCount > 0) Badge { Text("$cartCount") }
                        }) {
                            Icon(Icons.Default.ShoppingBag, contentDescription = "Warenkorb")
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface,
                ),
            )
        }
    ) { padding ->
        LazyVerticalGrid(
            columns = GridCells.Fixed(2),
            contentPadding = PaddingValues(
                top = padding.calculateTopPadding() + 8.dp,
                start = 12.dp, end = 12.dp, bottom = 16.dp,
            ),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            // Hero Banner
            item(span = { GridItemSpan(2) }) {
                HeroBanner()
            }

            // Suchfeld
            item(span = { GridItemSpan(2) }) {
                OutlinedTextField(
                    value = searchText,
                    onValueChange = { q ->
                        searchText = q
                        productVm.search(q)
                    },
                    placeholder = { Text("Schmuck suchen…") },
                    leadingIcon = { Icon(Icons.Default.Search, null) },
                    trailingIcon = {
                        if (searchText.isNotEmpty()) {
                            IconButton(onClick = { searchText = ""; productVm.search("") }) {
                                Icon(Icons.Default.Close, null)
                            }
                        }
                    },
                    singleLine = true,
                    shape = RoundedCornerShape(28.dp),
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            // Kategorie-Filter
            item(span = { GridItemSpan(2) }) {
                val filters = listOf(
                    null to "Alle",
                    "Earrings" to "Ohrringe",
                    "Necklaces" to "Halsketten",
                    "Rings" to "Ringe",
                    "Bracelets" to "Armbänder",
                )
                Row(
                    modifier = Modifier
                        .horizontalScroll(rememberScrollState())
                        .padding(vertical = 4.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    filters.forEach { (type, label) ->
                        val selected = state.filterType == type
                        FilterChip(
                            selected = selected,
                            onClick = { productVm.selectFilter(type) },
                            label = { Text(label) },
                            colors = FilterChipDefaults.filterChipColors(
                                selectedContainerColor = EstidGold,
                                selectedLabelColor = Color.White,
                            ),
                        )
                    }
                }
            }

            // Kollektionen
            if (state.collections.isNotEmpty()) {
                item(span = { GridItemSpan(2) }) {
                    Text(
                        "Kollektionen",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.padding(top = 8.dp, bottom = 4.dp),
                    )
                }
                item(span = { GridItemSpan(2) }) {
                    Row(
                        modifier = Modifier.horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        state.collections.forEach { col ->
                            val selected = state.selectedCollection == col.handle
                            CollectionChip(
                                title = col.title,
                                imageUrl = col.imageUrl,
                                selected = selected,
                                onClick = {
                                    productVm.selectCollection(if (selected) null else col.handle)
                                },
                            )
                        }
                    }
                }
            }

            // Produkte Header
            item(span = { GridItemSpan(2) }) {
                Row(
                    modifier = Modifier.padding(top = 12.dp, bottom = 4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        if (state.searchQuery.isNotBlank()) "Suchergebnisse" else "Produkte",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.weight(1f),
                    )
                    if (!state.isLoading) {
                        Text(
                            "${state.products.size} Artikel",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }

            // Lade-Indikator
            if (state.isLoading) {
                item(span = { GridItemSpan(2) }) {
                    Box(Modifier.fillMaxWidth().padding(40.dp), contentAlignment = Alignment.Center) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(12.dp)) {
                            CircularProgressIndicator(color = EstidGold)
                            Text("Produkte werden geladen…", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }

            // Fehler
            state.error?.let { err ->
                item(span = { GridItemSpan(2) }) {
                    Card(
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Warning, null, tint = MaterialTheme.colorScheme.error)
                            Spacer(Modifier.width(12.dp))
                            Column(Modifier.weight(1f)) {
                                Text("Verbindungsfehler", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                                Text(err, style = MaterialTheme.typography.bodySmall)
                            }
                            TextButton(onClick = { productVm.loadProducts() }) { Text("Erneut") }
                        }
                    }
                }
            }

            // Produktliste
            items(state.products, key = { it.id }) { product ->
                ProductCard(
                    product = product,
                    onClick = { onProductClick(product.handle) },
                )
            }

            // "Mehr laden"
            if (state.hasNextPage) {
                item(span = { GridItemSpan(2) }) {
                    Button(
                        onClick = { productVm.loadMore() },
                        modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
                        enabled = !state.isLoadingMore,
                        colors = ButtonDefaults.buttonColors(containerColor = EstidGold),
                    ) {
                        if (state.isLoadingMore) {
                            CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp, color = Color.White)
                        } else {
                            Text("Mehr laden")
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun HeroBanner() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(180.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(
                Brush.linearGradient(
                    colors = listOf(Color(0xFF1A1A1A), Color(0xFF3D2B1F))
                )
            ),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Text(
                "ESTID Fashion",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Light,
                color = Color(0xFFC9A84C),
                letterSpacing = androidx.compose.ui.unit.TextUnit(4f, androidx.compose.ui.unit.TextUnitType.Sp),
            )
            Spacer(Modifier.height(6.dp))
            Text(
                "Handgefertigter Premium-Schmuck",
                style = MaterialTheme.typography.bodySmall,
                color = Color.White.copy(alpha = 0.8f),
                textAlign = TextAlign.Center,
            )
        }
    }
}

@Composable
private fun CollectionChip(
    title: String,
    imageUrl: String?,
    selected: Boolean,
    onClick: () -> Unit,
) {
    Surface(
        shape = RoundedCornerShape(12.dp),
        color = if (selected) EstidGold else MaterialTheme.colorScheme.surfaceVariant,
        modifier = Modifier
            .width(100.dp)
            .clickable(onClick = onClick),
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(8.dp),
        ) {
            if (imageUrl != null) {
                AsyncImage(
                    model = imageUrl,
                    contentDescription = title,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.size(56.dp).clip(RoundedCornerShape(8.dp)),
                )
                Spacer(Modifier.height(4.dp))
            }
            Text(
                title,
                style = MaterialTheme.typography.labelSmall,
                textAlign = TextAlign.Center,
                color = if (selected) Color.White else MaterialTheme.colorScheme.onSurface,
                maxLines = 2,
            )
        }
    }
}
