package de.estidfashion.app.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

// ESTID Fashion – Farben (elegant, feminin, premium)
val EstidGold = Color(0xFFC9A84C)
val EstidGoldLight = Color(0xFFE8C97A)
val EstidDark = Color(0xFF1A1A1A)
val EstidSurface = Color(0xFFFAF8F5)
val EstidOnSurface = Color(0xFF2D2D2D)
val EstidBorder = Color(0xFFE8E0D8)
val EstidAccent = Color(0xFF8B6E47)

private val LightColorScheme = lightColorScheme(
    primary = EstidGold,
    onPrimary = Color.White,
    primaryContainer = Color(0xFFF5EDD8),
    onPrimaryContainer = EstidDark,
    secondary = EstidAccent,
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFEDE0D0),
    onSecondaryContainer = EstidDark,
    tertiary = Color(0xFF8B7355),
    background = EstidSurface,
    onBackground = EstidOnSurface,
    surface = Color.White,
    onSurface = EstidOnSurface,
    surfaceVariant = Color(0xFFF0EAE0),
    onSurfaceVariant = Color(0xFF5C5048),
    outline = EstidBorder,
    error = Color(0xFFBA1A1A),
)

private val DarkColorScheme = darkColorScheme(
    primary = EstidGoldLight,
    onPrimary = EstidDark,
    primaryContainer = Color(0xFF4A3B1F),
    onPrimaryContainer = EstidGoldLight,
    secondary = Color(0xFFD4A96A),
    background = Color(0xFF12100E),
    onBackground = Color(0xFFEDE5D8),
    surface = Color(0xFF1E1A16),
    onSurface = Color(0xFFEDE5D8),
    surfaceVariant = Color(0xFF2E2820),
    outline = Color(0xFF5C5048),
)

@Composable
fun ESTIDFashionTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = false,
    content: @Composable () -> Unit,
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val ctx = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(ctx) else dynamicLightColorScheme(ctx)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }
    MaterialTheme(colorScheme = colorScheme, typography = Typography, content = content)
}
