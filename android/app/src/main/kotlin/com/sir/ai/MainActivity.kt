package com.sir.ai

import android.app.Activity
import android.os.Bundle
import android.view.Gravity
import android.widget.TextView

/** Minimal launch screen for validating the APK/JNI build only; no model inference is performed. */
class MainActivity : Activity() {
    private external fun nativeBuildStatus(): String

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val message = buildString {
            appendLine("S.I.R.")
            appendLine()
            appendLine("Android build foundation")
            appendLine()
            append(nativeBuildStatus())
        }

        setContentView(
            TextView(this).apply {
                text = message
                textSize = 20f
                gravity = Gravity.CENTER
                setPadding(32, 32, 32, 32)
                contentDescription = message
            },
        )
    }

    companion object {
        init {
            System.loadLibrary("sir_android")
        }
    }
}
