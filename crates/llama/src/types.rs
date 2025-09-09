use async_openai::types::{
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessageContent,
};

pub use llama_cpp_2::model::LlamaChatMessage;

pub trait FromOpenAI {
    fn from_openai(message: &ChatCompletionRequestMessage) -> Self;
}

impl FromOpenAI for LlamaChatMessage {
    fn from_openai(message: &ChatCompletionRequestMessage) -> Self {
        match message {
            ChatCompletionRequestMessage::System(system) => {
                let content = match &system.content {
                    ChatCompletionRequestSystemMessageContent::Text(text) => text,
                    _ => todo!(),
                };

                LlamaChatMessage::new("system".into(), content.into()).unwrap()
            }
            ChatCompletionRequestMessage::Assistant(assistant) => {
                let content = match &assistant.content {
                    Some(ChatCompletionRequestAssistantMessageContent::Text(text)) => text,
                    _ => todo!(),
                };
                LlamaChatMessage::new("assistant".into(), content.into()).unwrap()
            }
            ChatCompletionRequestMessage::User(user) => {
                let content = match &user.content {
                    ChatCompletionRequestUserMessageContent::Text(text) => text,
                    _ => todo!(),
                };

                LlamaChatMessage::new("user".into(), content.into()).unwrap()
            }
            _ => todo!(),
        }
    }
}

pub struct LlamaRequest {
    pub grammar: Option<String>,
    pub messages: Vec<LlamaChatMessage>,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub num_ctx: u32,
    pub repeat_penalty: f32,
    pub model_name: Option<String>,
}

impl Default for LlamaRequest {
    fn default() -> Self {
        Self {
            grammar: None,
            messages: Vec::new(),
            temperature: 0.8,   // Default value
            top_k: 40,          // Default value
            top_p: 0.9,         // Default value
            min_p: 0.05,        // Default value
            num_ctx: 4096,      // Default value
            repeat_penalty: 1.1, // Default value
            model_name: None,
        }
    }
}
