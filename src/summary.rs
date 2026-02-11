use anyhow::{Context, Result};
use serde_json::json;
use std::fmt::Write;
use std::fs;

use crate::api::OpenRouterClient;
use crate::file_processor::ProcessedFile;
use crate::logger::Logger;

/// Combines file contents into a single document and computes word statistics.
pub struct SummaryInput {
    pub combined_text: String,
    pub total_words: usize,
    pub file_count: usize,
}

impl SummaryInput {
    /// Builds the combined text from processed files.
    pub fn from_files(files: &[ProcessedFile]) -> Self {
        let estimated_len: usize = files.iter().map(|f| f.name.len() + f.content.len() + 20).sum();
        let mut combined = String::with_capacity(estimated_len);
        let mut total_words: usize = 0;

        for file in files {
            total_words += file.content.split_whitespace().count();
            // Using `write!` avoids extra allocations compared to `format!` + `push_str`.
            let _ = write!(combined, "=== File: {} ===\n{}\n\n", file.name, file.content);
        }

        Self {
            combined_text: combined,
            total_words,
            file_count: files.len(),
        }
    }

    /// Calculates the target word count based on the compression percent.
    pub fn target_words(&self, compress_pct: u8) -> usize {
        let raw = (self.total_words as f64 * compress_pct as f64 / 100.0).ceil() as usize;
        raw.max(50)
    }
}

/// Generates a summary via OpenRouter and writes it to `output_path`.
pub async fn generate_and_save(
    api: &OpenRouterClient,
    model: &str,
    input: &SummaryInput,
    compress_pct: u8,
    output_path: &str,
    logger: &Logger,
) -> Result<()> {
    let target_words = input.target_words(compress_pct);

    logger.info(&format!(
        "Found {} file(s). Total words: {}. Target: ~{} words ({}%).",
        input.file_count, input.total_words, target_words, compress_pct
    ));
    logger.info("Sending to OpenRouter for summary...");

    let system_prompt = format!(
        "Ты — эксперт по составлению резюме и аналитических сводок. \
         Пользователь предоставит содержимое нескольких файлов. \
         Твоя задача — создать подробное резюме ВСЕХ предоставленных файлов в формате Markdown.\n\n\
         ВАЖНОЕ ОГРАНИЧЕНИЕ: Исходный текст содержит {total} слов. \
         Твоё резюме ДОЛЖНО содержать примерно {target} слов (около {pct}% от оригинала). \
         Считай слова внимательно. НЕ пиши значительно больше или меньше {target} слов.\n\n\
         Резюме должно включать:\n\
         - Главный заголовок\n\
         - Раздел с общим обзором\n\
         - Раздел для каждого файла с его ключевыми тезисами\n\
         - Заключение, объединяющее всё вместе\n\n\
         ОБЯЗАТЕЛЬНО: Отвечай ТОЛЬКО на русском языке. \
         Выводи ТОЛЬКО Markdown-резюме, без лишних комментариев.",
        total = input.total_words,
        target = target_words,
        pct = compress_pct,
    );

    let messages = vec![
        json!({"role": "system", "content": system_prompt}),
        json!({"role": "user", "content": format!(
            "Пожалуйста, составь резюме следующих файлов:\n\n{}", input.combined_text
        )}),
    ];

    let response = api
        .chat(model, messages)
        .await
        .context("Failed to get summary from OpenRouter")?;

    fs::write(output_path, &response)
        .with_context(|| format!("Failed to write summary to '{output_path}'"))?;

    logger.info(&format!("Summary successfully written to '{output_path}'"));

    Ok(())
}
