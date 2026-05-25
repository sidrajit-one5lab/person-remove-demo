// OpenCV 4.12 Android library module — Kotlin DSL replacement for the SDK's legacy
// build.gradle. AGP 9–compatible. Original is preserved as build.gradle.original
// for reference.
plugins {
    id("com.android.library")
}

android {
    namespace = "org.opencv"
    compileSdk = 36

    defaultConfig {
        minSdk = 21

        externalNativeBuild {
            cmake {
                arguments += listOf("-DANDROID_STL=c++_shared")
                targets += listOf("opencv_jni_shared")
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildTypes {
        debug {
            // OpenCV's CMake controls stripping.
            packaging { jniLibs.keepDebugSymbols += "**/*.so" }
        }
        release {
            packaging { jniLibs.keepDebugSymbols += "**/*.so" }
            isMinifyEnabled = false
        }
    }

    sourceSets {
        getByName("main") {
            jniLibs.setSrcDirs(listOf("sdk/native/libs"))
            java.setSrcDirs(listOf("sdk/java/src"))
            res.setSrcDirs(listOf("sdk/java/res"))
            manifest.srcFile("sdk/java/AndroidManifest.xml")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("sdk/libcxx_helper/CMakeLists.txt")
        }
    }

    buildFeatures {
        prefabPublishing = true
        buildConfig = true
    }

    prefab {
        create("opencv_jni_shared") {
            headers = "sdk/native/jni/include"
        }
    }
}
