mod api;
mod config;
mod file_processor;
mod logger;
mod summary;

use anyhow::{Context, Result};

use crate::api::OpenRouterClient;
use crate::config::Config;
use crate::file_processor::read_all_files;
use crate::logger::Logger;
use crate::summary::SummaryInput;

#[tokio::main]
async fn main() -> Result<()> {
    // .env is optional — environment variables may be set externally.
    let _ = dotenvy::dotenv();

    let config = Config::load("config.yaml")?;

    let logger = Logger::new(&config.log_file);

    let api_key =
        std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set in .env")?;

    let api = OpenRouterClient::new(api_key)?;

    logger.info(&format!(
        "Reading files from directory: '{}'",
        config.files_directory
    ));
    logger.info(&format!("OCR model: {}", config.ocr_model));
    logger.info(&format!("Summary model: {}", config.summary_model));

    let files = read_all_files(&config.files_directory, &api, &config.ocr_model, &logger).await?;

    if files.is_empty() {
        logger.warn(&format!("No files found in '{}'.", config.files_directory));
        return Ok(());
    }

    let input = SummaryInput::from_files(&files);

    summary::generate_and_save(
        &api,
        &config.summary_model,
        &input,
        config.compress_percent(),
        &config.output_path,
        &logger,
    )
    .await?;

    Ok(())
}