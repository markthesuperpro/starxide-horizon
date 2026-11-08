//! Process-wide switches for Android-only runtime policies.
use std::sync::atomic::{AtomicBool, Ordering};

/// Copper Mobile is enabled for this game process.
static COPPER_MOBILE: AtomicBool = AtomicBool::new(false);

/// Configure Copper Mobile before the worker pool and renderer are initialized.
pub fn configure_copper_mobile(enabled: bool) {
    COPPER_MOBILE.store(enabled, Ordering::Relaxed);
}

/// Whether Copper Mobile is enabled for this game process.
pub fn copper() -> bool {
    COPPER_MOBILE.load(Ordering::Relaxed)
}
