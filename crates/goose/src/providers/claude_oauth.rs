use anyhow::Result;
use base64::Engine;
use chrono::{DateTime, Utc};
use etcetera::{choose_app_strategy, AppStrategy};
use once_cell::sync::Lazy;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::Digest;
use std::{fs, io::Write, path::PathBuf};
use tokio::sync::Mutex as TokioMutex;

// OAuth configuration constants
const AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
const TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";
const REDIRECT_URL: &str = "https://console.anthropic.com/oauth/code/callback";
const CODE_CHALLENGE_METHOD: &str = "S256";

// Global mutex to ensure only one OAuth flow runs at a time
static OAUTH_MUTEX: Lazy<TokioMutex<()>> = Lazy::new(|| TokioMutex::new(()));

#[derive(Serialize, Deserialize)]
struct TokenData {
    /// The access token used to authenticate API requests
    access_token: String,

    /// Optional refresh token that can be used to obtain a new access token
    /// when the current one expires, enabling offline access without user interaction
    refresh_token: Option<String>,

    /// When the access token expires (if known)
    /// Used to determine when a token needs to be refreshed
    expires_at: Option<DateTime<Utc>>,
}

struct TokenCache {
    cache_path: PathBuf,
}

fn get_cache_dir() -> PathBuf {
    choose_app_strategy(crate::config::APP_STRATEGY.clone())
        .expect("goose requires a home dir")
        .in_config_dir("claude/oauth")
}

impl TokenCache {
    fn new(client_id: &str, scopes: &[String]) -> Self {
        let cache_key = Self::generate_cache_key(client_id, scopes);
        let cache_path = Self::get_cache_path(&cache_key);
        Self { cache_path }
    }

    fn generate_cache_key(client_id: &str, scopes: &[String]) -> String {
        let mut hasher = sha2::Sha256::new();
        hasher.update("console.anthropic.com".as_bytes());
        hasher.update(client_id.as_bytes());
        hasher.update(scopes.join(",").as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn get_cache_path(cache_key: &str) -> PathBuf {
        let cache_dir = get_cache_dir();
        fs::create_dir_all(&cache_dir).unwrap_or_else(|e| {
            tracing::warn!("Failed to create cache directory: {}", e);
        });
        cache_dir.join(format!("{}.json", cache_key))
    }

    fn load_token(&self) -> Option<TokenData> {
        match fs::read_to_string(&self.cache_path) {
            Ok(contents) => match serde_json::from_str::<TokenData>(&contents) {
                Ok(token_data) => {
                    // Only use tokens with refresh capability
                    if token_data.refresh_token.is_some() {
                        tracing::debug!("Loaded cached token");
                        Some(token_data)
                    } else {
                        tracing::debug!("Cached token has no refresh token, ignoring");
                        None
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to parse cached token: {}", e);
                    None
                }
            },
            Err(e) => {
                tracing::debug!("No cached token found: {}", e);
                None
            }
        }
    }

    fn save_token(&self, token_data: &TokenData) -> Result<()> {
        if let Some(parent) = self.cache_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let contents = serde_json::to_string(token_data)?;
        fs::write(&self.cache_path, contents)?;
        tracing::debug!("Saved token to cache");
        Ok(())
    }
}

struct ClaudeOAuthFlow {
    client_id: String,
    scopes: Vec<String>,
    state: String,
    verifier: String,
}

impl ClaudeOAuthFlow {
    fn new(client_id: String, scopes: Vec<String>) -> Self {
        let verifier = Self::generate_pkce_verifier();
        // Claude uses verifier as state (non-standard OAuth behavior)
        let state = verifier.clone();

        Self {
            client_id,
            scopes,
            state,
            verifier,
        }
    }

    fn generate_pkce_verifier() -> String {
        let mut random_bytes = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut random_bytes);
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&random_bytes)
    }

    fn get_authorization_url(&self) -> String {
        let challenge = self.generate_pkce_challenge();
        let scope = self.scopes.join(" ");
        
        let params = [
            ("code", "true"), // Claude-specific parameter - must be first!
            ("client_id", &self.client_id),
            ("response_type", "code"),
            ("redirect_uri", REDIRECT_URL),
            ("scope", &scope),
            ("code_challenge", &challenge),
            ("code_challenge_method", CODE_CHALLENGE_METHOD),
            ("state", &self.state),
        ];
        
        format!("{AUTHORIZE_URL}?{}", serde_urlencoded::to_string(params).unwrap())
    }

    fn generate_pkce_challenge(&self) -> String {
        let digest = sha2::Sha256::digest(self.verifier.as_bytes());
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest)
    }

    fn parse_code_from_callback(&self, input: &str) -> Result<(String, String)> {
        let input = input.trim();
        
        // Try parsing as URL first
        if let Ok((code, state)) = self.parse_as_url(input) {
            return Ok((code, state));
        }
        
        // Try parsing as code#state format
        if let Some(hash_pos) = input.find('#') {
            let code = input[..hash_pos].to_string();
            let state = input[hash_pos + 1..].to_string();
            return Ok((code, state));
        }
        
        // Fallback: assume it's just the code
        tracing::warn!("No state found in callback, using default state");
        Ok((input.to_string(), self.state.clone()))
    }

    fn parse_as_url(&self, input: &str) -> Result<(String, String)> {
        let url = url::Url::parse(input)?;
        
        let code = url
            .query_pairs()
            .find(|(key, _)| key == "code")
            .map(|(_, value)| value.to_string())
            .ok_or_else(|| anyhow::anyhow!("Code not found in URL"))?;
            
        let state = url.fragment()
            .unwrap_or(&self.state)
            .to_string();
            
        Ok((code, state))
    }

    async fn exchange_code_for_token(&self, code: &str, state: &str) -> Result<TokenData> {
        let payload = self.build_token_exchange_payload(code, state);
        
        tracing::debug!("Exchanging authorization code for token");
        
        let response = self.make_token_request(&payload).await?;
        self.extract_token_data(&response, None)
    }

    fn build_token_exchange_payload(&self, code: &str, state: &str) -> Value {
        serde_json::json!({
            "code": code,
            "state": state,
            "grant_type": "authorization_code",
            "client_id": &self.client_id,
            "redirect_uri": REDIRECT_URL,
            "code_verifier": &self.verifier,
        })
    }

    async fn refresh_token(&self, refresh_token: &str) -> Result<TokenData> {
        let payload = serde_json::json!({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": &self.client_id,
        });
        
        tracing::debug!("Refreshing Claude OAuth token");
        
        let response = self.make_token_request(&payload).await?;
        self.extract_token_data(&response, Some(refresh_token))
    }

    async fn make_token_request(&self, payload: &Value) -> Result<Value> {
        let client = reqwest::Client::new();
        let resp = client
            .post(TOKEN_URL)
            .header("Content-Type", "application/json")
            .json(payload)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let err_text = resp.text().await?;
            tracing::error!("Token request failed with status {}: {}", status, err_text);
            return Err(anyhow::anyhow!(
                "Token request failed with status {}: {}",
                status, err_text
            ));
        }

        Ok(resp.json().await?)
    }

    fn extract_token_data(
        &self,
        token_response: &Value,
        old_refresh_token: Option<&str>,
    ) -> Result<TokenData> {
        let access_token = self.extract_access_token(token_response)?;
        let refresh_token = self.extract_refresh_token(token_response, old_refresh_token);
        let expires_at = self.calculate_expiration(token_response);

        Ok(TokenData {
            access_token,
            refresh_token,
            expires_at,
        })
    }

    fn extract_access_token(&self, response: &Value) -> Result<String> {
        response
            .get("access_token")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("access_token not found in token response"))
    }

    fn extract_refresh_token(&self, response: &Value, old_refresh_token: Option<&str>) -> Option<String> {
        response
            .get("refresh_token")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| old_refresh_token.map(|s| s.to_string()))
    }

    fn calculate_expiration(&self, response: &Value) -> Option<DateTime<Utc>> {
        response
            .get("expires_in")
            .and_then(|v| v.as_u64())
            .map(|expires_in| {
                let expiration = Utc::now() + chrono::Duration::seconds(expires_in as i64);
                tracing::debug!("Token expires at: {}", expiration);
                expiration
            })
    }

    async fn execute(&self) -> Result<TokenData> {
        self.open_authorization_browser()?;
        let (code, state) = self.prompt_for_authorization_code()?;
        
        tracing::info!("Exchanging authorization code for token");
        self.exchange_code_for_token(&code, &state).await
    }

    fn open_authorization_browser(&self) -> Result<()> {
        let authorization_url = self.get_authorization_url();

        println!("\n🌐 Opening browser for Claude OAuth authentication...");
        println!("If the browser doesn't open automatically, please visit:");
        println!("{}\n", authorization_url);

        if let Err(e) = webbrowser::open(&authorization_url) {
            tracing::warn!("Failed to open browser: {}", e);
            println!("⚠️  Failed to open browser automatically.");
        }
        
        Ok(())
    }

    fn prompt_for_authorization_code(&self) -> Result<(String, String)> {
        println!("After authorizing in your browser, you'll be redirected to a page showing your authorization code.");
        println!("📋 Please copy and paste the authorization code below:");
        println!("(You can paste the full URL, the code#state format, or just the code)\n");

        print!("> ");
        std::io::stdout().flush()?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        self.parse_code_from_callback(&input)
    }
}

pub async fn get_claude_oauth_token(client_id: &str, scopes: &[String]) -> Result<String> {
    // Ensure only one OAuth flow runs at a time
    let _guard = OAUTH_MUTEX.lock().await;

    let token_cache = TokenCache::new(client_id, scopes);

    // Try to use cached token
    if let Some(token) = try_use_cached_token(&token_cache, client_id, scopes).await? {
        return Ok(token);
    }

    // No valid cached token, execute new OAuth flow
    tracing::info!("Starting new Claude OAuth flow");
    let flow = ClaudeOAuthFlow::new(client_id.to_string(), scopes.to_vec());
    let token = flow.execute().await?;

    // Cache and return
    token_cache.save_token(&token)?;
    Ok(token.access_token)
}

async fn try_use_cached_token(
    token_cache: &TokenCache,
    client_id: &str,
    scopes: &[String],
) -> Result<Option<String>> {
    let token = match token_cache.load_token() {
        Some(token) => token,
        None => return Ok(None),
    };

    // Check if token is still valid
    if is_token_valid(&token) {
        tracing::debug!("Using valid cached token");
        return Ok(Some(token.access_token));
    }

    // Try to refresh expired token
    if let Some(refresh_token) = token.refresh_token {
        match refresh_cached_token(token_cache, client_id, scopes, &refresh_token).await {
            Ok(new_token) => return Ok(Some(new_token)),
            Err(e) => {
                tracing::warn!("Failed to refresh token: {}", e);
            }
        }
    }

    Ok(None)
}

fn is_token_valid(token: &TokenData) -> bool {
    match token.expires_at {
        Some(expires_at) => expires_at > Utc::now(),
        None => {
            // No expiration info - assume valid and let server reject if expired
            tracing::debug!("Token has no expiration time, assuming valid");
            true
        }
    }
}

async fn refresh_cached_token(
    token_cache: &TokenCache,
    client_id: &str,
    scopes: &[String],
    refresh_token: &str,
) -> Result<String> {
    tracing::info!("Refreshing expired Claude token");
    
    let flow = ClaudeOAuthFlow::new(client_id.to_string(), scopes.to_vec());
    let new_token = flow.refresh_token(refresh_token).await?;
    
    token_cache.save_token(&new_token)?;
    tracing::info!("Successfully refreshed Claude token");
    
    Ok(new_token.access_token)
}






            
