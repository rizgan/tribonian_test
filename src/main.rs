use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct Config {
    files_directory: String,
    compress_summary: u8,
    ocr_model: String,
    summary_model: String,
}

const OPENROUTER_API_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// Sends a chat completion request to OpenRouter and returns the response text.
async fn openrouter_chat(
    client: &Client,
    api_key: &str,
    model: &str,
    messages: Vec<Value>,
) -> Result<String> {
    let body = json!({
        "model": model,
        "messages": messages,
    });

    let response = client
        .post(OPENROUTER_API_URL)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .context("Failed to send request to OpenRouter")?;

    let status = response.status();
    let response_text = response
        .text()
        .await
        .context("Failed to read response body")?;

    if !status.is_success() {
        anyhow::bail!("OpenRouter API error ({}): {}", status, response_text);
    }

    let response_json: Value =
        serde_json::from_str(&response_text).context("Failed to parse OpenRouter response")?;

    let content = response_json["choices"][0]["message"]["content"]
        .as_str()
        .context("No content in OpenRouter response")?
        .to_string();

    Ok(content)
}

/// Returns the MIME type for a given image file extension.
fn mime_type_for_image(ext: &str) -> &str {
    match ext {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        "tiff" | "tif" => "image/tiff",
        _ => "image/jpeg",
    }
}

/// Extracts text and description from an image using OpenRouter Vision API.
async fn extract_text_from_image(
    client: &Client,
    api_key: &str,
    model: &str,
    path: &Path,
) -> Result<String> {
    let bytes =
        fs::read(path).with_context(|| format!("Failed to read image: {}", path.display()))?;

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("jpeg")
        .to_lowercase();

    let mime = mime_type_for_image(&ext);
    let b64 = BASE64.encode(&bytes);
    let data_url = format!("data:{};base64,{}", mime, b64);

    let messages = vec![json!({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Проанализируй это изображение. \
                         Извлеки ВЕСЬ текст, который есть на изображении, сохраняя структуру и форматирование. \
                         Если на изображении есть графики, диаграммы, таблицы или другие визуальные элементы — \
                         опиши их содержимое и данные подробно. \
                         Отвечай на русском языке."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url
                }
            }
        ]
    })];

    openrouter_chat(client, api_key, model, messages).await
}

/// Extracts text from a PDF using OpenRouter's native PDF processing.
async fn extract_text_from_pdf(
    client: &Client,
    api_key: &str,
    model: &str,
    path: &Path,
) -> Result<String> {
    let bytes =
        fs::read(path).with_context(|| format!("Failed to read PDF: {}", path.display()))?;

    let b64 = BASE64.encode(&bytes);
    let data_url = format!("data:application/pdf;base64,{}", b64);

    let filename = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let messages = vec![json!({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Извлеки ВЕСЬ текст из этого PDF документа, сохраняя структуру и форматирование. \
                         Если в документе есть графики, диаграммы, таблицы или изображения — \
                         опиши их содержимое подробно. \
                         Отвечай на русском языке."
            },
            {
                "type": "file",
                "file": {
                    "filename": filename,
                    "file_data": data_url
                }
            }
        ]
    })];

    openrouter_chat(client, api_key, model, messages).await
}

/// File type classification.
enum FileType {
    Text,
    Pdf,
    Image,
}

/// Classifies a file by its extension.
fn classify_file(ext: &str) -> Option<FileType> {
    let text_extensions = [
        "txt", "md", "rs", "toml", "yaml", "yml", "json", "csv", "log", "cfg", "ini", "xml",
        "html", "css", "js", "ts", "py", "sh", "bat", "c", "cpp", "h", "hpp", "java", "go",
        "rb", "php", "sql", "r", "swift", "kt", "scala", "tex", "rtf",
    ];
    let image_extensions = ["jpg", "jpeg", "png", "webp", "gif", "bmp", "tiff", "tif"];

    if ext == "pdf" {
        Some(FileType::Pdf)
    } else if image_extensions.contains(&ext) {
        Some(FileType::Image)
    } else if text_extensions.contains(&ext) {
        Some(FileType::Text)
    } else {
        None
    }
}

/// Reads all files from the directory. Images and PDFs are processed via OpenRouter OCR.
async fn read_all_files(
    dir: &str,
    client: &Client,
    api_key: &str,
    ocr_model: &str,
) -> Result<Vec<(String, String)>> {
    let path = Path::new(dir);
    if !path.exists() {
        anyhow::bail!("Directory '{}' does not exist", dir);
    }

    let mut files_content: Vec<(String, String)> = Vec::new();

    for entry in fs::read_dir(path).context("Failed to read directory")? {
        let entry = entry?;
        let file_path = entry.path();

        if !file_path.is_file() {
            continue;
        }

        let ext = match file_path.extension().and_then(|e| e.to_str()) {
            Some(e) => e.to_lowercase(),
            None => continue,
        };

        let filename = file_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        match classify_file(&ext) {
            Some(FileType::Text) => match fs::read_to_string(&file_path) {
                Ok(content) => {
                    println!("  Read (text): {}", filename);
                    files_content.push((filename, content));
                }
                Err(e) => eprintln!("  Skipping '{}': {}", filename, e),
            },
            Some(FileType::Pdf) => {
                println!("  Processing (PDF via OpenRouter): {}...", filename);
                match extract_text_from_pdf(client, api_key, ocr_model, &file_path).await {
                    Ok(content) => {
                        println!("  Done: {}", filename);
                        files_content.push((filename, content));
                    }
                    Err(e) => eprintln!("  Skipping PDF '{}': {}", filename, e),
                }
            }
            Some(FileType::Image) => {
                println!("  Processing (image via OpenRouter): {}...", filename);
                match extract_text_from_image(client, api_key, ocr_model, &file_path).await {
                    Ok(content) => {
                        println!("  Done: {}", filename);
                        files_content.push((filename, content));
                    }
                    Err(e) => eprintln!("  Skipping image '{}': {}", filename, e),
                }
            }
            None => {}
        }
    }

    Ok(files_content)
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().context("Failed to load .env file")?;

    let config_content =
        fs::read_to_string("config.yaml").context("Failed to read config.yaml")?;
    let config: Config =
        serde_yaml::from_str(&config_content).context("Failed to parse config.yaml")?;

    let api_key =
        std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set in .env")?;

    let client = Client::new();

    println!(
        "Reading files from directory: '{}'",
        config.files_directory
    );
    println!("OCR model: {}", config.ocr_model);
    println!("Summary model: {}", config.summary_model);
    println!();

    let files =
        read_all_files(&config.files_directory, &client, &api_key, &config.ocr_model).await?;

    if files.is_empty() {
        println!("No files found in '{}'.", config.files_directory);
        return Ok(());
    }

    let compress = config.compress_summary.clamp(1, 100);

    let mut combined = String::new();
    let mut total_words: usize = 0;
    for (name, content) in &files {
        total_words += content.split_whitespace().count();
        combined.push_str(&format!("=== File: {} ===\n{}\n\n", name, content));
    }

    let target_words = ((total_words as f64) * (compress as f64) / 100.0).ceil() as usize;
    let target_words = target_words.max(50);

    println!(
        "Found {} file(s). Total words: {}. Target: ~{} words ({}%).",
        files.len(),
        total_words,
        target_words,
        compress
    );
    println!("Sending to OpenRouter for summary...\n");

    let system_prompt = format!(
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
    );

    let messages = vec![
        json!({"role": "system", "content": system_prompt}),
        json!({"role": "user", "content": format!("Пожалуйста, составь резюме следующих файлов:\n\n{}", combined)}),
    ];

    let response = openrouter_chat(&client, &api_key, &config.summary_model, messages)
        .await
        .context("Failed to get summary from OpenRouter")?;

    let output_path = "summary.md";
    fs::write(output_path, &response).context("Failed to write summary.md")?;

    println!("Summary successfully written to '{}'", output_path);

    Ok(())
}