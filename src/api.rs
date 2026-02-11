use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;

const OPENROUTER_API_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// Wrapper around the OpenRouter API.
pub struct OpenRouterClient {
    client: Client,
    api_key: String,
}

impl OpenRouterClient {
    /// Creates a new client with the given API key and sensible defaults.
    pub fn new(api_key: String) -> Result<Self> {
        let client = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .context("Failed to build HTTP client")?;

        Ok(Self { client, api_key })
    }

    /// Sends a chat completion request and returns the response content.
    pub async fn chat(&self, model: &str, messages: Vec<Value>) -> Result<String> {
        let body = json!({
            "model": model,
            "messages": messages,
        });

        let response = self
            .client
            .post(OPENROUTER_API_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
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
            anyhow::bail!("OpenRouter API error ({status}): {response_text}");
        }

        let response_json: Value =
            serde_json::from_str(&response_text).context("Failed to parse OpenRouter response")?;

        response_json["choices"][0]["message"]["content"]
            .as_str()
            .map(String::from)
            .context("No content in OpenRouter response")
    }
}
