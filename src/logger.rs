use chrono::Local;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::Mutex;

/// Log level for messages.
#[derive(Clone, Copy)]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

impl LogLevel {
    fn as_str(self) -> &'static str {
        match self {
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
        }
    }
}

/// A simple logger that writes to both stdout/stderr and an optional log file.
pub struct Logger {
    file: Option<Mutex<File>>,
}

impl Logger {
    /// Creates a new logger. If `log_path` is non-empty, log messages are also
    /// appended to the specified file. If the file cannot be opened, logging
    /// continues to the console only.
    pub fn new(log_path: &str) -> Self {
        let file = if log_path.is_empty() {
            None
        } else {
            match OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_path)
            {
                Ok(f) => {
                    println!("Logging to file: {log_path}");
                    Some(Mutex::new(f))
                }
                Err(e) => {
                    eprintln!("Warning: could not open log file '{log_path}': {e}");
                    None
                }
            }
        };

        Self { file }
    }

    /// Logs a message at the given level to console and (optionally) to the file.
    pub fn log(&self, level: LogLevel, message: &str) {
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");
        let level_str = level.as_str();
        let formatted = format!("[{timestamp}] [{level_str}] {message}");

        // Console output
        match level {
            LogLevel::Error => eprintln!("{formatted}"),
            _ => println!("{formatted}"),
        }

        // File output
        if let Some(ref file_mutex) = self.file {
            if let Ok(mut f) = file_mutex.lock() {
                let _ = writeln!(f, "{formatted}");
            }
        }
    }

    /// Convenience: log at INFO level.
    pub fn info(&self, message: &str) {
        self.log(LogLevel::Info, message);
    }

    /// Convenience: log at WARN level.
    pub fn warn(&self, message: &str) {
        self.log(LogLevel::Warn, message);
    }

    /// Convenience: log at ERROR level.
    pub fn error(&self, message: &str) {
        self.log(LogLevel::Error, message);
    }
}
