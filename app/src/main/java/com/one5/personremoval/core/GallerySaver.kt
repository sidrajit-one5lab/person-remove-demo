package com.one5.personremoval.core

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Saves a JPEG to the device's gallery in the "PersonRemoval" sub-album.
 *
 * On API 29+ we use MediaStore with RELATIVE_PATH (scoped storage, no permission
 * needed). On API 28 and below we write directly to Pictures/PersonRemoval/ and
 * notify MediaStore — caller must have WRITE_EXTERNAL_STORAGE for that path.
 */
object GallerySaver {

    private const val TAG = "GallerySaver"
    private const val ALBUM = "PersonRemoval"

    fun save(context: Context, jpegBytes: ByteArray): Uri? {
        val displayName = "PersonRemoval_${timestamp()}.jpg"
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                saveScoped(context, jpegBytes, displayName)
            } else {
                saveLegacy(context, jpegBytes, displayName)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "save failed", t)
            null
        }
    }

    private fun saveScoped(context: Context, bytes: ByteArray, name: String): Uri? {
        val cv = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(
                MediaStore.Images.Media.RELATIVE_PATH,
                Environment.DIRECTORY_PICTURES + "/" + ALBUM
            )
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv)
            ?: return null

        // If openOutputStream returns null, or write() throws partway, we own a
        // pending row pointing at a (possibly partial) file. Delete it before
        // returning/rethrowing so the gallery doesn't accumulate orphans.
        try {
            resolver.openOutputStream(uri)?.use { it.write(bytes) }
                ?: run {
                    resolver.delete(uri, null, null)
                    return null
                }
        } catch (t: Throwable) {
            resolver.delete(uri, null, null)
            throw t
        }
        cv.clear()
        cv.put(MediaStore.Images.Media.IS_PENDING, 0)
        resolver.update(uri, cv, null, null)
        Log.i(TAG, "saved: $uri")
        return uri
    }

    private fun saveLegacy(context: Context, bytes: ByteArray, name: String): Uri? {
        val dir = File(
            @Suppress("DEPRECATION")
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
            ALBUM
        )
        if (!dir.exists() && !dir.mkdirs()) {
            Log.e(TAG, "couldn't create $dir")
            return null
        }
        val file = File(dir, name)
        FileOutputStream(file).use { it.write(bytes) }

        // Tell MediaStore there's a new image so it shows in Gallery.
        val cv = ContentValues().apply {
            put(MediaStore.Images.Media.DATA, file.absolutePath)
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
        }
        val uri = context.contentResolver
            .insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv)
        Log.i(TAG, "saved (legacy): ${file.absolutePath}")
        return uri
    }

    private fun timestamp(): String =
        SimpleDateFormat("yyyy-MM-dd_HHmmss", Locale.US).format(Date())
}
