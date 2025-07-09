use anyhow::Result;
use async_trait::async_trait;
use axum::http::HeaderMap;
use regex::Regex;
use reqwest::{Client, StatusCode};
use serde::Serialize;
use serde_json::{Value, json};
use std::time::Duration;

use super::base::{Provider, ProviderMetadata, ProviderUsage};
use super::claude_oauth;
use super::errors::ProviderError;
use super::formats::anthropic::{get_usage, response_to_message};
use super::utils::{emit_debug_trace, get_model};

use crate::message::Message;
use crate::model::ModelConfig;
use crate::providers::anthropic::ANTHROPIC_DOC_URL;
use crate::providers::base::ModelInfo;
use crate::providers::formats::anthropic::create_request_multi_system;
use mcp_core::tool::Tool;

pub const CLAUDE_DEFAULT_MODEL: &str = "claude-sonnet-4-20250514";
pub const CLAUDE_KNOWN_MODELS: &[&str] = &[
    "claude-3-7-sonnet-20250219",
    "claude-3-5-haiku-20241022",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
];

const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const SCOPES: &[&str] = &["org:create_api_key", "user:profile", "user:inference"];
const API_BASE_URL: &str = "https://api.anthropic.com";

// HTTP headers as constants for better maintainability
const HEADER_AUTHORIZATION: &str = "Authorization";
const HEADER_ANTHROPIC_VERSION: &str = "anthropic-version";
const HEADER_ANTHROPIC_BETA: &str = "anthropic-beta";

// Header values as constants
const ANTHROPIC_VERSION_VALUE: &str = "2023-06-01";
const ANTHROPIC_BETA_VALUE: &str = "oauth-2025-04-20,interleaved-thinking-2025-05-14";

#[derive(Debug, Serialize)]
pub struct ClaudeProvider {
    #[serde(skip)]
    client: Client,
    model: ModelConfig,
}

impl Default for ClaudeProvider {
    fn default() -> Self {
        let model = ModelConfig::new(ClaudeProvider::metadata().default_model);
        ClaudeProvider::from_env(model).expect("Failed to initialize Claude provider")
    }
}

impl ClaudeProvider {
    pub fn from_env(model: ModelConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .build()?;

        Ok(Self { client, model })
    }

    async fn ensure_auth_header(&self) -> Result<String> {
        tracing::debug!("Getting OAuth token for Claude provider");
        let scopes = SCOPES.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let token = claude_oauth::get_claude_oauth_token(CLIENT_ID, &scopes).await?;
        tracing::debug!("Successfully obtained OAuth token");
        Ok(format!("Bearer {}", token))
    }

    /// Check if the latest message contains "ultrathink" and modify payload accordingly
    fn check_and_add_thinking_mode(messages: &[Message], payload: &mut Value) {
        if let Some(latest_message) = messages.last() {
            let text_content = latest_message.as_concat_text();

            // Case-insensitive search for "ultrathink"
            let re = Regex::new(r"(?i)ultrathink").unwrap();
            if re.is_match(&text_content) {
                tracing::info!("Found 'ultrathink' in the latest message");

                // Add thinking configuration to the payload
                if let Some(payload_obj) = payload.as_object_mut() {
                    payload_obj.insert(
                        "thinking".to_string(),
                        json!({
                            "type": "enabled",
                            "budget_tokens": 31999
                        }),
                    );
                    tracing::debug!("Added thinking configuration to payload");
                }
            }
        }
    }

    async fn post(&self, headers: HeaderMap, payload: Value) -> Result<Value, ProviderError> {
        let url = format!("{}/v1/messages?beta=true", API_BASE_URL);

        tracing::debug!("Making request to Claude API: {}", url);

        let response = self
            .client
            .post(&url)
            .headers(headers)
            .json(&payload)
            .send()
            .await?;

        let status = response.status();
        let payload: Option<Value> = response.json().await.ok();

        tracing::debug!("Claude API response status: {}", status);

        // https://docs.anthropic.com/en/api/errors
        match status {
            StatusCode::OK => {
                tracing::debug!("Claude API request successful");
                payload.ok_or_else(|| {
                    tracing::error!("Claude API returned OK but body is not valid JSON");
                    ProviderError::RequestFailed("Response body is not valid JSON".to_string())
                })
            }
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                tracing::error!("Claude authentication failed with status: {}", status);
                Err(ProviderError::Authentication(format!(
                    "Authentication failed. Please ensure your OAuth tokens are valid. \
                    Status: {}. Response: {:?}",
                    status, payload
                )))
            }
            StatusCode::BAD_REQUEST => {
                let mut error_msg = "Unknown error".to_string();
                if let Some(payload) = &payload {
                    if let Some(error) = payload.get("error") {
                        tracing::debug!("Bad Request Error: {error:?}");
                        error_msg = error
                            .get("message")
                            .and_then(|m| m.as_str())
                            .unwrap_or("Unknown error")
                            .to_string();
                        if error_msg.to_lowercase().contains("too long")
                            || error_msg.to_lowercase().contains("too many")
                        {
                            return Err(ProviderError::ContextLengthExceeded(
                                error_msg.to_string(),
                            ));
                        }
                    }
                }
                tracing::debug!(
                    "{}",
                    format!(
                        "Provider request failed with status: {}. Payload: {:?}",
                        status, payload
                    )
                );
                Err(ProviderError::RequestFailed(format!(
                    "Request failed with status: {}. Message: {}",
                    status, error_msg
                )))
            }
            StatusCode::TOO_MANY_REQUESTS => {
                tracing::warn!("Claude API rate limit exceeded");
                Err(ProviderError::RateLimitExceeded(format!("{:?}", payload)))
            }
            StatusCode::INTERNAL_SERVER_ERROR | StatusCode::SERVICE_UNAVAILABLE => {
                tracing::error!("Claude API server error: {}", status);
                Err(ProviderError::ServerError(format!("{:?}", payload)))
            }
            _ => {
                tracing::debug!(
                    "{}",
                    format!(
                        "Provider request failed with status: {}. Payload: {:?}",
                        status, payload
                    )
                );
                Err(ProviderError::RequestFailed(format!(
                    "Request failed with status: {}",
                    status
                )))
            }
        }
    }
}

#[async_trait]
impl Provider for ClaudeProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::with_models(
            "claude",
            "Claude",
            "Access Claude models via OAuth authentication",
            CLAUDE_DEFAULT_MODEL,
            CLAUDE_KNOWN_MODELS
                .iter()
                .map(|&name| ModelInfo::new(name, 200_000))
                .collect::<Vec<_>>(),
            ANTHROPIC_DOC_URL,
            vec![], // No configuration needed - OAuth is handled automatically
        )
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    #[tracing::instrument(
        skip(self, system, messages, tools),
        fields(model_config, input, output, input_tokens, output_tokens, total_tokens)
    )]
    async fn complete(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        tracing::debug!(
            "Starting Claude completion request with {} messages and {} tools",
            messages.len(),
            tools.len()
        );

        let mut multi_system_prompts = vec![];

        // We need this system prompt otherwise higher models wouldn't work.
        if !self.model.model_name.starts_with("claude-3-5-haiku-") {
            multi_system_prompts.push("You are Claude Code, Anthropic's official CLI for Claude.");
        }
        if !system.is_empty() {
            multi_system_prompts.push(system);
        }

        let mut payload =
            create_request_multi_system(&self.model, &multi_system_prompts, messages, tools)?;

        // Check for "ultrathink" and add thinking mode if needed
        Self::check_and_add_thinking_mode(messages, &mut payload);

        // Build headers
        let mut headers = HeaderMap::new();
        let auth_header = self.ensure_auth_header().await?;
        headers.insert(HEADER_AUTHORIZATION, auth_header.parse().unwrap());
        headers.insert(HEADER_ANTHROPIC_VERSION, ANTHROPIC_VERSION_VALUE.parse().unwrap());
        headers.insert(HEADER_ANTHROPIC_BETA, ANTHROPIC_BETA_VALUE.parse().unwrap());

        // Make request
        let response = self.post(headers, payload.clone()).await?;

        // Parse response
        let message = response_to_message(response.clone())?;
        let usage = get_usage(&response)?;
        let model = get_model(&response);
        emit_debug_trace(&self.model, &payload, &response, &usage);

        tracing::info!(
            "Claude completion successful - tokens: input={:?}, output={:?}, total={:?}",
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens
        );

        Ok((message, ProviderUsage::new(model, usage)))
    }
}
