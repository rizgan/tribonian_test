use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub files_directory: String,
    pub compress_summary: u8,
    pub ocr_model: String,
    pub summary_model: String,
    /// Output file path for the generated summary (defaults to "summary.md").
    #[serde(default = "default_output_path")]
    pub output_path: String,
    /// Log file path (defaults to "app.log"). Set to empty string to disable file logging.
    #[serde(default = "default_log_file")]
    pub log_file: String,
}

fn default_output_path() -> String {
    "summary.md".to_string()
}

fn default_log_file() -> String {
    "app.log".to_string()
}

impl Config {
    /// Loads configuration from a YAML file at the given path.
    pub fn load(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {path}"))?;
        let config: Config =
            serde_yaml::from_str(&content).context("Failed to parse config YAML")?;
        Ok(config)
    }

    /// Returns `compress_summary` clamped to [1, 100].
    pub fn compress_percent(&self) -> u8 {
        self.compress_summary.clamp(1, 100)
    }
}
