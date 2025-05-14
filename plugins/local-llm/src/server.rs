use std::net::{Ipv4Addr, SocketAddr};
use std::pin::Pin;

use axum::{
    extract::State as AxumState,
    http::StatusCode,
    response::{sse, IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use futures_util::StreamExt;
use tower_http::cors::{self, CorsLayer};

use async_openai::types::{
    ChatChoice, ChatChoiceStream, ChatCompletionResponseMessage, ChatCompletionStreamResponseDelta,
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
    Role,
};

#[derive(Clone)]
pub struct ServerHandle {
    pub addr: SocketAddr,
    pub shutdown: tokio::sync::watch::Sender<()>,
}

impl ServerHandle {
    pub fn shutdown(self) -> Result<(), tokio::sync::watch::error::SendError<()>> {
        self.shutdown.send(())
    }
}

pub async fn run_server(model_manager: crate::ModelManager) -> Result<ServerHandle, crate::Error> {
    tracing::info!("Starting local LLM server initialization");
    
    // Create router with routes
    let app = Router::new()
        .route("/health", get(health))
        .route("/chat/completions", post(chat_completions))
        .with_state(model_manager)
        .layer(
            CorsLayer::new()
                .allow_origin(cors::Any)
                .allow_methods(cors::Any)
                .allow_headers(cors::Any),
        );

    // Try to bind to a random port on localhost
    tracing::info!("Attempting to bind TCP listener");
    let listener = match tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).await {
        Ok(l) => {
            tracing::info!("TCP listener bound successfully");
            l
        }
        Err(e) => {
            tracing::error!("Failed to bind TCP listener: {}", e);
            return Err(e.into());
        }
    };

    // Get the server address
    let server_addr = match listener.local_addr() {
        Ok(addr) => {
            tracing::info!("Server will listen on: {}", addr);
            addr
        }
        Err(e) => {
            tracing::error!("Failed to get local address: {}", e);
            return Err(e.into());
        }
    };

    let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(());

    let server_handle = ServerHandle {
        addr: server_addr,
        shutdown: shutdown_tx,
    };

    tokio::spawn(async move {
        tracing::info!("Starting axum server on {}", server_addr);
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                tracing::info!("Waiting for shutdown signal");
                shutdown_rx.changed().await.ok();
                tracing::info!("Shutdown signal received, stopping server");
            })
            .await
            .unwrap_or_else(|e| {
                tracing::error!("Server error: {}", e);
            });
    });

    tracing::info!("Local LLM server started successfully at {}", server_addr);
    Ok(server_handle)
}

async fn health(AxumState(model_manager): AxumState<crate::ModelManager>) -> impl IntoResponse {
    match model_manager.get_model().await {
        Ok(_) => {
            tracing::info!("Health check passed");
            StatusCode::OK
        },
        Err(e) => {
            tracing::error!("Health check failed: {}", e);
            StatusCode::SERVICE_UNAVAILABLE
        },
    };
}

async fn chat_completions(
    AxumState(model_manager): AxumState<crate::ModelManager>,
    Json(request): Json<CreateChatCompletionRequest>,
) -> Result<Response, (StatusCode, String)> {
    let model = model_manager
        .get_model()
        .await
        .map_err(|e| {
            tracing::error!("Failed to get model: {}", e);
            (StatusCode::SERVICE_UNAVAILABLE, e.to_string())
        })?;

    // Get current model info from model manager
    let current_model_info = model_manager.get_current_model();
    
    tracing::info!("Processing chat completion with model: {}", 
        if let Some(model) = &current_model_info {
            model.model_name()
        } else {
            "unknown"
        });

    let res = inference_with_hypr(&model, &request)
        .await
        .map_err(|e| {
            tracing::error!("Inference error: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        })?;

    Ok(res.into_response())
}

async fn inference_with_hypr(
    model: &hypr_llama::Llama,
    request: &CreateChatCompletionRequest,
) -> Result<impl IntoResponse, crate::Error> {
    #[allow(deprecated)]
    let empty_message = ChatCompletionResponseMessage {
        content: None,
        refusal: None,
        tool_calls: None,
        role: Role::Assistant,
        audio: None,
        function_call: None,
    };

    let empty_choice = ChatChoice {
        message: empty_message.clone(),
        index: 0,
        finish_reason: None,
        logprobs: None,
    };

    let empty_response = CreateChatCompletionResponse {
        id: uuid::Uuid::new_v4().to_string(),
        choices: vec![],
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32,
        model: request.model.clone(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion".to_string(),
        usage: None,
    };

    let empty_stream_response = CreateChatCompletionStreamResponse {
        id: empty_response.id.clone(),
        choices: vec![],
        created: empty_response.created,
        model: empty_response.model.clone(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion.chunk".to_string(),
        usage: None,
    };

    #[allow(deprecated)]
    let empty_stream_response_delta = ChatCompletionStreamResponseDelta {
        content: None,
        function_call: None,
        tool_calls: None,
        role: None,
        refusal: None,
    };

    let is_stream = request.stream.unwrap_or(false);

    if !is_stream {
        let completion =
            futures_util::StreamExt::collect::<String>(build_response(model, request)?).await;

        let res = CreateChatCompletionResponse {
            choices: vec![ChatChoice {
                message: ChatCompletionResponseMessage {
                    content: Some(completion),
                    ..empty_message
                },
                ..empty_choice
            }],
            ..empty_response
        };

        return Ok(Json(res).into_response());
    }

    let res = if request.model == "mock-onboarding" {
        build_mock_response()
    } else {
        build_response(model, request)?
    };

    let stream = res
        .map(move |chunk| CreateChatCompletionStreamResponse {
            choices: vec![ChatChoiceStream {
                index: 0,
                delta: ChatCompletionStreamResponseDelta {
                    content: Some(chunk),
                    ..empty_stream_response_delta.clone()
                },
                finish_reason: None,
                logprobs: None,
            }],
            ..empty_stream_response.clone()
        })
        .map(|chunk| {
            Ok::<_, std::convert::Infallible>(
                sse::Event::default().data(serde_json::to_string(&chunk).unwrap()),
            )
        });

    Ok(sse::Sse::new(stream).into_response())
}

fn build_response(
    model: &hypr_llama::Llama,
    request: &CreateChatCompletionRequest,
) -> Result<Pin<Box<dyn futures_util::Stream<Item = String> + Send>>, crate::Error> {
    let messages = request
        .messages
        .iter()
        .map(hypr_llama::FromOpenAI::from_openai)
        .collect();

    // Determine model type based on the request
    let model_type = if request.model.to_lowercase().contains("qwen") {
        "qwen"
    } else {
        "llama"
    };
    
    tracing::info!("Using model type: {} for inference", model_type);
    tracing::info!("Request details: stream={}, temperature={}", 
        request.stream.unwrap_or(false),
        request.temperature.unwrap_or(0.7));

    // Build request with model-specific parameters
    let llama_request = if model_type == "qwen" {
        tracing::info!("Using Qwen-specific parameters");
        hypr_llama::LlamaRequest {
            messages,
            grammar: Some(hypr_gbnf::GBNF::Enhance(Some(vec!["".to_string()])).build()),
            temperature: 0.6,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            num_ctx: 32768,
            repeat_penalty: 1.2,
            model_name: Some("qwen".to_string()),
        }
    } else {
        // Default parameters for other models
        tracing::info!("Using default model parameters");
        hypr_llama::LlamaRequest {
            messages,
            grammar: Some(hypr_gbnf::GBNF::Enhance(Some(vec!["".to_string()])).build()),
            ..Default::default()
        }
    };
    
    tracing::debug!("Configured parameters: temp={}, top_k={}, top_p={}, min_p={}, ctx={}, repeat_penalty={}",
        llama_request.temperature,
        llama_request.top_k,
        llama_request.top_p,
        llama_request.min_p,
        llama_request.num_ctx,
        llama_request.repeat_penalty);

    // Get the raw stream from the model
    tracing::info!("Starting model generation stream");
    let raw_stream = match model.generate_stream(llama_request) {
        Ok(stream) => {
            tracing::info!("Model stream created successfully");
            stream
        },
        Err(e) => {
            tracing::error!("Failed to create model stream: {}", e);
            return Err(crate::Error::HyprLlamaError(e));
        }
    };
    
    // Process the stream to filter out <thinking> tags for Qwen model
    if model_type == "qwen" {
        tracing::info!("Applying thinking tag filter for Qwen model");
        tracing::debug!("Stream will be processed to remove <thinking> tags");
        Ok(Box::pin(filter_thinking_tags(raw_stream)))
    } else {
        tracing::info!("Using raw stream without filtering");
        Ok(Box::pin(raw_stream))
    }
}

// Filter out content between <thinking> and </thinking> tags
fn filter_thinking_tags<S>(stream: S) -> impl futures_util::Stream<Item = String>
where
    S: futures_util::Stream<Item = String> + Send + 'static,
{
    use futures_util::StreamExt;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    tracing::info!("Initializing thinking tag filter");
    
    // State to track tag presence across chunks
    let state = Arc::new(Mutex::new(FilterState {
        in_thinking_tag: false,
        buffer: String::new(),
    }));

    stream.filter_map(move |chunk| {
        let state_clone = state.clone();
        async move {
            tracing::debug!("Processing stream chunk of length {}", chunk.len());
            let mut state = state_clone.lock().await;
            
            // Add current chunk to buffer
            state.buffer.push_str(&chunk);
            
            // Process the buffer
            let (output, remaining) = process_thinking_tags(&state.buffer, state.in_thinking_tag);
            
            // Update state
            state.buffer = remaining;
            
            // Update the state based on tag presence
            let contains_closing = state.buffer.contains("</thinking>");
            let contains_opening = state.buffer.contains("<thinking>");
            
            // If we found a closing tag, ensure we're no longer in a thinking tag
            if contains_closing {
                state.in_thinking_tag = false;
                tracing::debug!("Found closing thinking tag");
            }
            
            // If we found an opening tag, mark that we're in a thinking tag
            if contains_opening {
                state.in_thinking_tag = true;
                tracing::debug!("Found opening thinking tag");
            }
            
            // Return processed output if not empty
            if !output.is_empty() {
                tracing::debug!("Returning filtered output of length {}", output.len());
                Some(output)
            } else {
                tracing::debug!("No output to return after filtering");
                None
            }
        }
    })
}

struct FilterState {
    in_thinking_tag: bool,
    buffer: String,
}

// Process a text buffer to remove content between <thinking> and </thinking> tags
fn process_thinking_tags(text: &str, already_in_tag: bool) -> (String, String) {
    let mut result = String::new();
    let mut remaining = String::new();
    let mut in_thinking_tag = already_in_tag;
    let mut current_pos = 0;
    
    if already_in_tag {
        tracing::debug!("Starting processing inside thinking tag");
    }
    
    // Find all occurrences of opening and closing tags
    while current_pos < text.len() {
        if in_thinking_tag {
            // We're inside a thinking tag, look for closing tag
            if let Some(end_pos) = text[current_pos..].find("</thinking>") {
                // Skip everything until after the closing tag
                tracing::debug!("Found closing tag at position {}", current_pos + end_pos);
                current_pos += end_pos + "</thinking>".len();
                in_thinking_tag = false;
            } else {
                // No closing tag found, save the rest for later processing
                tracing::debug!("No closing tag found, buffering {} chars", text[current_pos..].len());
                remaining = text[current_pos..].to_string();
                break;
            }
        } else {
            // We're outside a thinking tag, look for opening tag
            if let Some(start_pos) = text[current_pos..].find("<thinking>") {
                // Add text before the tag
                tracing::debug!("Found opening tag at position {}", current_pos + start_pos);
                result.push_str(&text[current_pos..current_pos + start_pos]);
                current_pos += start_pos + "<thinking>".len();
                in_thinking_tag = true;
            } else {
                // No more tags, add remaining text
                tracing::debug!("No more tags, adding remaining {} chars", text[current_pos..].len());
                result.push_str(&text[current_pos..]);
                break;
            }
        }
    }
    
    // If we're at the end and there's no remaining text to process
    if current_pos >= text.len() {
        remaining.clear();
    }
    
    tracing::debug!("Processing result: {} chars output, {} chars remaining, in_tag={}", 
                   result.len(), remaining.len(), in_thinking_tag);
    
    (result, remaining)
}

fn build_mock_response() -> Pin<Box<dyn futures_util::Stream<Item = String> + Send>> {
    use futures_util::stream::{self, StreamExt};
    use std::time::Duration;

    let content = crate::ONBOARDING_ENHANCED_MD;
    let chunk_size = 30;

    let chunks = content
        .chars()
        .collect::<Vec<_>>()
        .chunks(chunk_size)
        .map(|c| c.iter().collect::<String>())
        .collect::<Vec<_>>();

    Box::pin(stream::iter(chunks).then(|chunk| async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        chunk
    }))
}
