from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class DataConfig:
    data_dir: Path = field(default_factory=lambda: Path("/home/luci/android-apks-dataset/ground_truth"))
    min_syscalls: int = 100

    @property
    def syscall_dir(self) -> Path:
        return self.data_dir / "syscall_traces"

    @property
    def sbom_dir(self) -> Path:
        return self.data_dir / "sboms"


@dataclass
class FeatureConfig:
    path_max_features: int = 300
    path_ngram_range: Tuple[int, int] = (1, 2)
    path_min_df: int = 5
    path_max_df: float = 0.8

    thread_max_features: int = 100
    thread_min_df: int = 3

    thread_syscall_max_features: int = 200
    thread_syscall_min_df: int = 5

    syscall_bigram_max_features: int = 100
    syscall_bigram_min_df: int = 5

    path_syscall_max_features: int = 50
    path_syscall_min_df: int = 5

    top_thread_syscalls: int = 100
    max_syscall_bigrams: int = 1000
    max_path_syscalls: int = 500


@dataclass
class ModelConfig:
    max_iter: int = 1000
    solver: str = "liblinear"
    random_state: int = 42


@dataclass
class EvalConfig:
    cv_folds: int = 5
    random_state: int = 42
    tier1_threshold: float = 0.70
    tier1_max_std: float = 0.15
    tier2_threshold: float = 0.40


TARGET_LIBRARIES: Dict[str, List[str]] = {
    "Flutter": ["io.flutter", "Flutter"],
    "WorkManager": ["androidx.work", "WorkManager"],
    "Room": ["androidx.room", "Room"],
    "Coroutines": ["kotlinx.coroutines", "Kotlin Coroutines"],
    "OkHttp": ["okhttp3", "OkHttp"],
    "Glide": ["com.bumptech.glide", "Glide"],
    "ExoPlayer": ["com.google.android.exoplayer", "androidx.media3", "ExoPlayer", "Media3"],
    "RxJava": ["io.reactivex", "RxJava"],
    "DataStore": ["androidx.datastore", "DataStore"],
    "Retrofit": ["retrofit2", "Retrofit"],
    "Realm": ["io.realm", "Realm"],
    "Lottie": ["com.airbnb.lottie", "Lottie"],
    "Fresco": ["com.facebook.fresco", "Fresco"],
    "Coil": ["coil", "io.coil"],
    "ReactNative": ["com.facebook.react", "React Native"],
    "Firebase": ["com.google.firebase", "Firebase"],
    "Sentry": ["io.sentry", "Sentry"],
    "Timber": ["timber.log", "Timber"],
    "Gson": ["com.google.gson", "Gson"],
    "Dagger": ["dagger", "com.google.dagger", "Dagger", "Hilt"],
    "Navigation": ["androidx.navigation", "Navigation"],
    "Paging": ["androidx.paging", "Paging"],
    "Lifecycle": ["androidx.lifecycle", "Lifecycle"],
    "Compose": ["androidx.compose", "Jetpack Compose"],
}

NOISE_PATTERNS: List[str] = [
    r"goldfish",
    r"logcat",
    r"^/proc/",
    r"^/sys/",
]
