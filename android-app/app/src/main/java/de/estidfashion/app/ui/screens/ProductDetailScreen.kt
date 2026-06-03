package de.estidfashion.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import coil.compose.AsyncImage
import de.estidfashion.app.data.models.CartItem
import de.estidfashion.app.ui.theme.EstidGold
import de.estidfashion.app.viewmodel.CartViewModel
import de.estidfashion.app.viewmodel.ProductDetailViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ProductDetailScreen(
    handle: String,
    onBack: () -> Unit,
    onCartClick: () -> Unit,
    detailVm: ProductDetailViewModel = hiltViewModel(),
    cartVm: CartViewModel = hiltViewModel(),
) {
    LaunchedEffect(handle) { detailVm.load(handle) }

    val product by detailVm.product.collectAsStateWithLifecycle()
    val isLoading by detailVm.isLoading.collectAsStateWithLifecycle()
    val selectedVariantIndex by detailVm.selectedVariantIndex.collectAsStateWithLifecycle()
    val cartCount by cartVm.itemCount.collectAsStateWithLifecycle()
    var addedToCart by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(product?.title ?: "", maxLines = 1, overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, "Zurück")
                    }
                },
                actions = {
                    IconButton(onClick = onCartClick) {
                        BadgedBox(badge = { if (cartCount > 0) Badge { Text("$cartCount") } }) {
                            Icon(Icons.Default.ShoppingBag, "Warenkorb")
                        }
                    }
                },
            )
        },
        bottomBar = {
            product?.let { p ->
                val variant = p.variants.getOrNull(selectedVariantIndex) ?: p.variants.firstOrNull()
                Surface(shadowElevation = 8.dp) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp)
                            .navigationBarsPadding(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(
                                "%.2f €".format(variant?.price ?: p.priceMin),
                                style = MaterialTheme.typography.headlineSmall,
                                fontWeight = FontWeight.Bold,
                                color = EstidGold,
                            )
                            variant?.compareAtPrice?.let {
                                Text(
                                    "%.2f €".format(it),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.outline,
                                    textDecoration = androidx.compose.ui.text.style.TextDecoration.LineThrough,
                                )
                            }
                        }
                        Button(
                            onClick = {
                                variant?.let { v ->
                                    cartVm.addToCart(
                                        CartItem(
                                            variantId = v.id,
                                            productId = p.id,
                                            title = p.title,
                                            variantTitle = v.title,
                                            price = v.price,
                                            imageUrl = p.featuredImageUrl,
                                            quantity = 1,
                                        )
                                    )
                                    addedToCart = true
                                }
                            },
                            enabled = variant?.availableForSale == true,
                            colors = ButtonDefaults.buttonColors(containerColor = EstidGold),
                            modifier = Modifier.height(50.dp),
                        ) {
                            Icon(Icons.Default.ShoppingCart, null, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(8.dp))
                            Text(
                                if (variant?.availableForSale == true) "In den Warenkorb" else "Ausverkauft",
                                style = MaterialTheme.typography.labelLarge,
                            )
                        }
                    }
                }
            }
        }
    ) { padding ->
        if (isLoading) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator(color = EstidGold)
            }
            return@Scaffold
        }

        val p = product ?: return@Scaffold

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(top = padding.calculateTopPadding())
                .verticalScroll(rememberScrollState()),
        ) {
            // Bildergalerie
            val images = p.images.ifEmpty { listOfNotNull(p.featuredImageUrl) }
            if (images.isNotEmpty()) {
                val pagerState = rememberPagerState(pageCount = { images.size })
                Box {
                    HorizontalPager(state = pagerState) { page ->
                        AsyncImage(
                            model = images[page],
                            contentDescription = p.title,
                            contentScale = ContentScale.Crop,
                            modifier = Modifier
                                .fillMaxWidth()
                                .aspectRatio(1f)
                                .background(MaterialTheme.colorScheme.surfaceVariant),
                        )
                    }
                    // Punkte-Indikator
                    if (images.size > 1) {
                        Row(
                            Modifier
                                .align(Alignment.BottomCenter)
                                .padding(bottom = 12.dp),
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                        ) {
                            repeat(images.size) { i ->
                                Box(
                                    Modifier
                                        .size(if (pagerState.currentPage == i) 8.dp else 6.dp)
                                        .clip(CircleShape)
                                        .background(if (pagerState.currentPage == i) EstidGold else Color.White.copy(alpha = 0.6f)),
                                )
                            }
                        }
                    }
                }
            }

            Column(Modifier.padding(16.dp)) {
                // Titel + Marke
                Text(p.productType, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Spacer(Modifier.height(4.dp))
                Text(p.title, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                Spacer(Modifier.height(4.dp))
                Text("von ${p.vendor}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)

                // Verfügbarkeit
                Spacer(Modifier.height(12.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    val v = p.variants.getOrNull(selectedVariantIndex)
                    Icon(
                        if (v?.availableForSale == true) Icons.Default.CheckCircle else Icons.Default.Cancel,
                        null,
                        tint = if (v?.availableForSale == true) Color(0xFF34A853) else MaterialTheme.colorScheme.error,
                        modifier = Modifier.size(16.dp),
                    )
                    Spacer(Modifier.width(4.dp))
                    Text(
                        if (v?.availableForSale == true) "Auf Lager" else "Ausverkauft",
                        style = MaterialTheme.typography.bodySmall,
                        color = if (v?.availableForSale == true) Color(0xFF34A853) else MaterialTheme.colorScheme.error,
                    )
                    v?.quantityAvailable?.takeIf { it in 1..10 }?.let {
                        Spacer(Modifier.width(8.dp))
                        Text("(nur $it übrig)", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
                    }
                }

                // Varianten-Auswahl (falls mehrere)
                if (p.variants.size > 1) {
                    Spacer(Modifier.height(16.dp))
                    Text("Ausführung wählen:", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.SemiBold)
                    Spacer(Modifier.height(8.dp))
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        p.variants.forEachIndexed { i, v ->
                            val selected = i == selectedVariantIndex
                            Surface(
                                shape = RoundedCornerShape(8.dp),
                                color = if (selected) EstidGold else MaterialTheme.colorScheme.surfaceVariant,
                                modifier = Modifier
                                    .clickable { detailVm.selectVariant(i) }
                                    .border(
                                        1.dp,
                                        if (selected) EstidGold else MaterialTheme.colorScheme.outline,
                                        RoundedCornerShape(8.dp),
                                    ),
                            ) {
                                Text(
                                    v.title,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = if (selected) Color.White else MaterialTheme.colorScheme.onSurface,
                                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                                )
                            }
                        }
                    }
                }

                // Beschreibung
                Spacer(Modifier.height(20.dp))
                HorizontalDivider()
                Spacer(Modifier.height(16.dp))
                Text("Beschreibung", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Spacer(Modifier.height(8.dp))
                Text(
                    p.description.replace(Regex("<[^>]*>"), ""),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    lineHeight = MaterialTheme.typography.bodyMedium.lineHeight,
                )

                // Tags
                if (p.tags.isNotEmpty()) {
                    Spacer(Modifier.height(16.dp))
                    HorizontalDivider()
                    Spacer(Modifier.height(12.dp))
                    Text("Tags", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.SemiBold)
                    Spacer(Modifier.height(8.dp))
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        p.tags.take(8).forEach { tag ->
                            SuggestionChip(
                                onClick = {},
                                label = { Text(tag, style = MaterialTheme.typography.labelSmall) },
                            )
                        }
                    }
                }

                // Versand-Info
                Spacer(Modifier.height(16.dp))
                HorizontalDivider()
                Spacer(Modifier.height(12.dp))
                ShippingInfoCard()
                Spacer(Modifier.height(padding.calculateBottomPadding() + 80.dp))
            }
        }
    }

    // Snackbar bei "In den Warenkorb"
    if (addedToCart) {
        LaunchedEffect(addedToCart) {
            kotlinx.coroutines.delay(2000)
            addedToCart = false
        }
        Snackbar(
            modifier = Modifier.padding(16.dp),
            action = {
                TextButton(onClick = { onCartClick(); addedToCart = false }) {
                    Text("Zum Warenkorb", color = EstidGold)
                }
            }
        ) { Text("Zum Warenkorb hinzugefügt") }
    }
}

@Composable
private fun ShippingInfoCard() {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        shape = RoundedCornerShape(12.dp),
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Versand & Rückgabe", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.SemiBold)
            InfoRow(Icons.Default.LocalShipping, "Versand nach Deutschland: 4,95 €")
            InfoRow(Icons.Default.Star, "Kostenloser Versand ab 50 €")
            InfoRow(Icons.Default.Replay, "30 Tage Rückgaberecht")
            InfoRow(Icons.Default.Security, "Sichere Zahlung · SSL verschlüsselt")
        }
    }
}

@Composable
private fun InfoRow(icon: androidx.compose.ui.graphics.vector.ImageVector, text: String) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Icon(icon, null, modifier = Modifier.size(16.dp), tint = EstidGold)
        Text(text, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}
