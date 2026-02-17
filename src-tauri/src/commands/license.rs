use crate::license::{LicenseState, LicenseStatus};
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::time::Instant;
use tauri::AppHandle;

#[derive(Clone, Debug)]
pub struct CachedLicense {
    pub status: LicenseStatus,
    cached_at: Instant,
}

impl CachedLicense {
    const CACHE_DURATION: std::time::Duration = std::time::Duration::from_secs(6 * 60 * 60);

    pub fn new(status: LicenseStatus) -> Self {
        Self {
            status,
            cached_at: Instant::now(),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.cached_at.elapsed() < Self::CACHE_DURATION
    }

    pub fn age(&self) -> std::time::Duration {
        self.cached_at.elapsed()
    }
}

impl UnwindSafe for CachedLicense {}
impl RefUnwindSafe for CachedLicense {}

fn self_hosted_status() -> LicenseStatus {
    LicenseStatus {
        status: LicenseState::Licensed,
        trial_days_left: None,
        license_type: Some("self-hosted".to_string()),
        license_key: Some("self-hosted".to_string()),
        expires_at: None,
    }
}

#[tauri::command]
pub async fn check_license_status(_app: AppHandle) -> Result<LicenseStatus, String> {
    log::info!("License check bypassed (self-hosted mode)");
    Ok(self_hosted_status())
}

pub async fn check_license_status_internal(_app: &AppHandle) -> Result<LicenseStatus, String> {
    Ok(self_hosted_status())
}

#[tauri::command]
pub async fn activate_license(
    _license_key: String,
    _app: AppHandle,
) -> Result<LicenseStatus, String> {
    log::info!("License activation bypassed (self-hosted mode)");
    Ok(self_hosted_status())
}

#[tauri::command]
pub async fn restore_license(_app: AppHandle) -> Result<LicenseStatus, String> {
    log::info!("License restore bypassed (self-hosted mode)");
    Ok(self_hosted_status())
}

#[tauri::command]
pub async fn deactivate_license(_app: AppHandle) -> Result<(), String> {
    log::info!("License deactivation bypassed (self-hosted mode)");
    Ok(())
}

#[tauri::command]
pub async fn open_purchase_page() -> Result<(), String> {
    log::info!("Opening purchase page");

    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg("https://voicetypr.com/#pricing")
            .spawn()
            .map_err(|e| format!("Failed to open browser: {}", e))?;
    }

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;

        std::process::Command::new("cmd")
            .args(&["/C", "start", "https://voicetypr.com/#pricing"])
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
            .map_err(|e| format!("Failed to open browser: {}", e))?;
    }

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg("https://voicetypr.com/#pricing")
            .spawn()
            .map_err(|e| format!("Failed to open browser: {}", e))?;
    }

    Ok(())
}

#[tauri::command]
pub async fn invalidate_license_cache(_app: AppHandle) -> Result<(), String> {
    Ok(())
}
