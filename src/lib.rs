//! 7q9k route: a local OpenAI, Ollama and Anthropic compatible proxy for the
//! free OpenCode models.

#[macro_use]
pub mod util;
pub mod assets;
pub mod convert;
pub mod log;
pub mod paths;
pub mod upstream;
pub mod config;
pub mod integrations;
pub mod metrics;
pub mod pricing;
pub mod server;
