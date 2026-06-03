package de.estidfashion.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.navigation.NavType
import androidx.navigation.compose.*
import androidx.navigation.navArgument
import dagger.hilt.android.AndroidEntryPoint
import de.estidfashion.app.ui.screens.*
import de.estidfashion.app.ui.theme.ESTIDFashionTheme

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            ESTIDFashionTheme {
                Surface(Modifier.fillMaxSize()) {
                    val navController = rememberNavController()

                    NavHost(
                        navController = navController,
                        startDestination = "home",
                    ) {
                        composable("home") {
                            HomeScreen(
                                onProductClick = { handle ->
                                    navController.navigate("product/$handle")
                                },
                                onCartClick = {
                                    navController.navigate("cart")
                                },
                            )
                        }

                        composable(
                            route = "product/{handle}",
                            arguments = listOf(navArgument("handle") { type = NavType.StringType }),
                        ) { backStack ->
                            val handle = backStack.arguments?.getString("handle") ?: return@composable
                            ProductDetailScreen(
                                handle = handle,
                                onBack = { navController.popBackStack() },
                                onCartClick = { navController.navigate("cart") },
                            )
                        }

                        composable("cart") {
                            CartScreen(
                                onBack = { navController.popBackStack() },
                            )
                        }
                    }
                }
            }
        }
    }
}
