package de.estidfashion.app.data.repository

import androidx.room.*
import de.estidfashion.app.data.models.CartItem
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject
import javax.inject.Singleton

@Dao
interface CartDao {
    @Query("SELECT * FROM cart_items")
    fun observeAll(): Flow<List<CartItem>>

    @Query("SELECT * FROM cart_items WHERE variantId = :variantId")
    suspend fun findByVariant(variantId: String): CartItem?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(item: CartItem)

    @Update
    suspend fun update(item: CartItem)

    @Delete
    suspend fun delete(item: CartItem)

    @Query("DELETE FROM cart_items")
    suspend fun clearAll()

    @Query("SELECT COUNT(*) FROM cart_items")
    fun observeCount(): Flow<Int>

    @Query("SELECT SUM(price * quantity) FROM cart_items")
    fun observeTotal(): Flow<Double?>
}

@Database(entities = [CartItem::class], version = 1, exportSchema = false)
abstract class AppDatabase : RoomDatabase() {
    abstract fun cartDao(): CartDao
}

@Singleton
class CartRepository @Inject constructor(private val dao: CartDao) {

    val cartItems: Flow<List<CartItem>> = dao.observeAll()
    val itemCount: Flow<Int> = dao.observeCount()
    val total: Flow<Double?> = dao.observeTotal()

    suspend fun addToCart(item: CartItem) {
        val existing = dao.findByVariant(item.variantId)
        if (existing != null) {
            dao.update(existing.copy(quantity = existing.quantity + item.quantity))
        } else {
            dao.insert(item)
        }
    }

    suspend fun updateQuantity(variantId: String, quantity: Int) {
        val item = dao.findByVariant(variantId) ?: return
        if (quantity <= 0) dao.delete(item) else dao.update(item.copy(quantity = quantity))
    }

    suspend fun removeItem(variantId: String) {
        dao.findByVariant(variantId)?.let { dao.delete(it) }
    }

    suspend fun clearCart() = dao.clearAll()
}
