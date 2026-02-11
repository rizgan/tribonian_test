use anyhow::{Context, Result};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::deepseek;
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct Config {
    files_directory: String,
    compress_summary: u8, // процент от исходного объёма (например, 25 = оставить 25%)
}

/// Extracts all text content from a PDF file using pdf_extract.
fn extract_text_from_pdf(path: &Path) -> Result<String> {
    let bytes = fs::read(path)
        .with_context(|| format!("Failed to read PDF file: {}", path.display()))?;

    let text = pdf_extract::extract_text_from_mem(&bytes)
        .with_context(|| format!("Failed to extract text from PDF: {}", path.display()))?;

    let text = text.trim().to_string();
    if text.is_empty() {
        anyhow::bail!("No text content found in PDF: {}", path.display());
    }

    Ok(text)
}

/// Reads all text files and PDFs from the specified directory and returns their content.
fn read_text_files(dir: &str) -> Result<Vec<(String, String)>> {
    let text_extensions = [
        "txt", "md", "rs", "toml", "yaml", "yml", "json", "csv", "log", "cfg", "ini", "xml",
        "html", "css", "js", "ts", "py", "sh", "bat", "c", "cpp", "h", "hpp", "java", "go",
        "rb", "php", "sql", "r", "swift", "kt", "scala", "tex", "rtf",
    ];

    let path = Path::new(dir);
    if !path.exists() {
        anyhow::bail!("Directory '{}' does not exist", dir);
    }

    let mut files_content: Vec<(String, String)> = Vec::new();

    for entry in fs::read_dir(path).context("Failed to read directory")? {
        let entry = entry?;
        let file_path = entry.path();

        if file_path.is_file() {
            if let Some(ext) = file_path.extension().and_then(|e| e.to_str()) {
                let ext_lower = ext.to_lowercase();

                if ext_lower == "pdf" {
                    // Handle PDF files
                    let filename = file_path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    match extract_text_from_pdf(&file_path) {
                        Ok(content) => {
                            println!("  Read (PDF): {}", filename);
                            files_content.push((filename, content));
                        }
                        Err(e) => {
                            eprintln!("  Skipping PDF '{}': {}", file_path.display(), e);
                        }
                    }
                } else if text_extensions.contains(&ext_lower.as_str()) {
                    // Handle text files
                    match fs::read_to_string(&file_path) {
                        Ok(content) => {
                            let filename = file_path
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string();
                            println!("  Read: {}", filename);
                            files_content.push((filename, content));
                        }
                        Err(e) => {
                            eprintln!(
                                "  Skipping '{}': {}",
                                file_path.display(),
                                e
                            );
                        }
                    }
                }
            }
        }
    }

    Ok(files_content)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file
    dotenvy::dotenv().context("Failed to load .env file")?;

    // Load config
    let config_content =
        fs::read_to_string("config.yaml").context("Failed to read config.yaml")?;
    let config: Config =
        serde_yaml::from_str(&config_content).context("Failed to parse config.yaml")?;

    println!("Reading files (text + PDF) from directory: '{}'", config.files_directory);

    // Read all text files
    let files = read_text_files(&config.files_directory)?;

    if files.is_empty() {
        println!("No files found in '{}'.", config.files_directory);
        return Ok(());
    }

    let compress = config.compress_summary.clamp(1, 100);

    // Build combined content and count words
    let mut combined = String::new();
    let mut total_words: usize = 0;
    for (name, content) in &files {
        total_words += content.split_whitespace().count();
        combined.push_str(&format!("=== File: {} ===\n{}\n\n", name, content));
    }

    let target_words = ((total_words as f64) * (compress as f64) / 100.0).ceil() as usize;
    let target_words = target_words.max(50); // минимум 50 слов

    println!(
        "\nFound {} text file(s). Total words: {}. Target: ~{} words ({}%). Sending to DeepSeek...\n",
        files.len(),
        total_words,
        target_words,
        compress
    );

    // Create DeepSeek client (reads DEEPSEEK_API_KEY from env)
    let client = deepseek::Client::from_env();

    let agent = client
        .agent(deepseek::DEEPSEEK_CHAT)
        .preamble(&format!(
            "Ты — эксперт по составлению резюме и аналитических сводок. \
             Пользователь предоставит содержимое нескольких файлов. \
             Твоя задача — создать подробное резюме ВСЕХ предоставленных файлов в формате Markdown.\n\n\
             ВАЖНОЕ ОГРАНИЧЕНИЕ: Исходный текст содержит {total_words} слов. \
             Твоё резюме ДОЛЖНО содержать примерно {target_words} слов (около {compress}% от оригинала). \
             Считай слова внимательно. НЕ пиши значительно больше или меньше {target_words} слов.\n\n\
             Резюме должно включать:\n\
             - Главный заголовок\n\
             - Раздел с общим обзором\n\
             - Раздел для каждого файла с его ключевыми тезисами\n\
             - Заключение, объединяющее всё вместе\n\n\
             ОБЯЗАТЕЛЬНО: Отвечай ТОЛЬКО на русском языке. \
             Выводи ТОЛЬКО Markdown-резюме, без лишних комментариев.",
            total_words = total_words,
            target_words = target_words,
            compress = compress
        ))
        .build();

    let prompt = format!(
        "Пожалуйста, составь резюме следующих файлов:\n\n{}",
        combined
    );

    let response = agent.prompt(&prompt).await.context("Failed to get response from DeepSeek")?;

    // Write summary to file
    let output_path = "summary.md";
    fs::write(output_path, &response).context("Failed to write summary.md")?;

    println!("Summary successfully written to '{}'", output_path);

    Ok(())
}