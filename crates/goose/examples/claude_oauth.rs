use anyhow::Result;
use dotenv::dotenv;
use goose::{
    message::Message,
    providers::{base::Provider, claude::ClaudeProvider},
};

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables from .env file
    dotenv().ok();

    println!("Claude OAuth Example");
    println!("===================");
    println!("\nThis example demonstrates Claude's OAuth authentication flow.");
    println!("Unlike token-based authentication, OAuth provides secure access");
    println!("without requiring you to manually create API keys.\n");

    // Create the provider - OAuth is the only authentication method
    let provider = ClaudeProvider::default();

    println!("Creating a request to Claude...\n");

    // Create a simple message
    let message = Message::user().with_text("Ultrathink, Tell me a short joke about programming.");

    // Get a response
    // Note: The first time you run this, it will open a browser for OAuth authentication
    // Subsequent runs will use the cached access token until it expires
    let (response, usage) = provider
        .complete(
            "Ultrathink",
            &[message],
            &[],
        )
        .await?;

    // Print the response and usage statistics
    println!("\nResponse from Claude:");
    println!("-------------------");
    for content in response.content {
        if let Some(text) = content.as_text() {
            println!("{}", text);
        } else {
            dbg!(content);
        }
    }
    println!("\nToken Usage:");
    println!("------------");
    println!("Input tokens: {:?}", usage.usage.input_tokens);
    println!("Output tokens: {:?}", usage.usage.output_tokens);
    println!("Total tokens: {:?}", usage.usage.total_tokens);

    println!("\n✅ OAuth authentication successful!");
    println!("\nNote: Your access token has been cached for future use.");
    println!("To force re-authentication, delete ~/.config/goose/claude/oauth/");

    let provider = ClaudeProvider::from_env(goose::model::ModelConfig::new(
        "claude-3-5-haiku-20241022".to_string(),
    ))?;

    // Create a simple message
    let message = Message::user().with_text("Tell me a short joke about programming.");

    // Get a response
    // Note: The first time you run this, it will open a browser for OAuth authentication
    // Subsequent runs will use the cached access token until it expires
    let (response, usage) = provider
        .complete(
            "You are an agentic assistant.",
            &[message],
            &[],
        )
        .await?;

    // Print the response and usage statistics
    println!("\nResponse from Claude Haiku:");
    println!("-------------------");
    for content in response.content {
        if let Some(text) = content.as_text() {
            println!("{}", text);
        } else {
            dbg!(content);
        }
    }
    println!("\nToken Usage:");
    println!("------------");
    println!("Input tokens: {:?}", usage.usage.input_tokens);
    println!("Output tokens: {:?}", usage.usage.output_tokens);
    println!("Total tokens: {:?}", usage.usage.total_tokens);



    Ok(())
}

