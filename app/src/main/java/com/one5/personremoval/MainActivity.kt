package com.one5.personremoval

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import com.one5.personremoval.core.NativeSession
import com.one5.personremoval.ui.CaptureScreen
import com.one5.personremoval.ui.theme.PersonRemovalTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        // Step 6a sanity check: prove the native library loads and OpenCV is linked.
        Log.i("PRNative", "OpenCV via JNI: ${NativeSession.openCvVersion()}")
        setContent {
            PersonRemovalTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    CaptureScreen()
                }
            }
        }
    }
}
