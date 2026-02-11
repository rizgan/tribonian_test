use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

use crate::api::OpenRouterClient;
use crate::logger::Logger;

// ---------------------------------------------------------------------------
// File type classification
// ---------------------------------------------------------------------------

const TEXT_EXTENSIONS: &[&str] = &[
    "txt", "md", "rs", "toml", "yaml", "yml", "json", "csv", "log", "cfg", "ini", "xml", "html", "css", "js", "ts", "py", "sh", "bat", "c", "cpp", "h", "hpp", "java", "go", "rb", "php", "sql", "r", "swift", "kt", "scala", "tex", "rtf",
];

const IMAGE_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "webp", "gif", "bmp", "tiff", "tif",
];

enum FileType {
    Text,
    Pdf,
    Image,
}

fn classify_file(ext: &str) -> Option<FileType> {
    if ext == "pdf" {
        Some(FileType::Pdf)
    } else if IMAGE_EXTENSIONS.contains(&ext) {
        Some(FileType::Image)
    } else if TEXT_EXTENSIONS.contains(&ext) {
        Some(FileType::Text)
    } else {
        None
    }
}

/// Returns the MIME type for a given image file extension.
fn mime_type_for_image(ext: &str) -> &'static str {
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

// ---------------------------------------------------------------------------
// Individual file extractors
// ---------------------------------------------------------------------------

/// Reads and base64-encodes a file, returning `(base64_string, extension)`.
fn read_and_encode(path: &Path) -> Result<(String, String)> {
    let bytes =
        fs::read(path).with_context(|| format!("Failed to read file: {}", path.display()))?;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("bin")
        .to_lowercase();
    Ok((BASE64.encode(&bytes), ext))
}

fn filename_of(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

/// Builds the OpenRouter messages payload for image OCR.
fn build_image_messages(data_url: &str) -> Vec<Value> {
    vec![json!({
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
                "image_url": { "url": data_url }
            }
        ]
    })]
}

/// Builds the OpenRouter messages payload for PDF extraction.
fn build_pdf_messages(filename: &str, data_url: &str) -> Vec<Value> {
    vec![json!({
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
    })]
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A processed file: name + extracted content.
pub struct ProcessedFile {
    pub name: String,
    pub content: String,
}

/// Reads all supported files from `dir`, using `api` + `ocr_model` for images/PDFs.
pub async fn read_all_files(
    dir: &str,
    api: &OpenRouterClient,
    ocr_model: &str,
    logger: &Logger,
) -> Result<Vec<ProcessedFile>> {
    let dir_path = Path::new(dir);
    if !dir_path.exists() {
        anyhow::bail!("Directory '{dir}' does not exist");
    }

    let mut results: Vec<ProcessedFile> = Vec::new();

    let mut entries: Vec<_> = fs::read_dir(dir_path)
        .context("Failed to read directory")?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let file_path = entry.path();

        if !file_path.is_file() {
            continue;
        }

        let ext = match file_path.extension().and_then(|e| e.to_str()) {
            Some(e) => e.to_lowercase(),
            None => continue,
        };

        let filename = filename_of(&file_path);

        let content = match classify_file(&ext) {
            Some(FileType::Text) => match fs::read_to_string(&file_path) {
                Ok(c) => {
                    logger.info(&format!("  Read (text): {filename}"));
                    c
                }
                Err(e) => {
                    logger.error(&format!("  Skipping '{filename}': {e}"));
                    continue;
                }
            },
            Some(FileType::Pdf) => {
                logger.info(&format!("  Processing (PDF via API): {filename}..."));
                match process_pdf(&file_path, api, ocr_model).await {
                    Ok(c) => {
                        logger.info(&format!("  Done: {filename}"));
                        c
                    }
                    Err(e) => {
                        logger.error(&format!("  Skipping PDF '{filename}': {e}"));
                        continue;
                    }
                }
            }
            Some(FileType::Image) => {
                logger.info(&format!("  Processing (image via API): {filename}..."));
                match process_image(&file_path, api, ocr_model).await {
                    Ok(c) => {
                        logger.info(&format!("  Done: {filename}"));
                        c
                    }
                    Err(e) => {
                        logger.error(&format!("  Skipping image '{filename}': {e}"));
                        continue;
                    }
                }
            }
            None => {
                logger.warn(&format!("  Skipping unsupported file: {filename}"));
                continue;
            }
        };

        results.push(ProcessedFile {
            name: filename,
            content,
        });
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

async fn process_image(path: &Path, api: &OpenRouterClient, model: &str) -> Result<String> {
    let (b64, ext) = read_and_encode(path)?;
    let mime = mime_type_for_image(&ext);
    let data_url = format!("data:{mime};base64,{b64}");
    api.chat(model, build_image_messages(&data_url)).await
}

async fn process_pdf(path: &Path, api: &OpenRouterClient, model: &str) -> Result<String> {
    let (b64, _) = read_and_encode(path)?;
    let data_url = format!("data:application/pdf;base64,{b64}");
    let filename = filename_of(path);
    api.chat(model, build_pdf_messages(&filename, &data_url))
        .await
}
