pub static SUPPORTED_MODELS: &[SupportedModel; 2] = &[
    SupportedModel::Qwen3_8b_Thinking,
    SupportedModel::Llama3p2_3bQ4,
];

#[derive(serde::Serialize, serde::Deserialize, specta::Type, Debug, Clone)]
#[allow(non_camel_case_types)]
pub enum SupportedModel {
    Qwen3_8b_Thinking,
    Llama3p2_3bQ4,
}

impl SupportedModel {
    pub fn model_path(&self, data_dir: impl Into<std::path::PathBuf>) -> std::path::PathBuf {
        match self {
            SupportedModel::Qwen3_8b_Thinking => data_dir.into().join("llm.gguf"),
            SupportedModel::Llama3p2_3bQ4 => data_dir.into().join("llm.gguf"),
        }
    }

    pub fn model_url(&self) -> &str {
        match self {
            SupportedModel::Qwen3_8b_Thinking => "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-UD-Q4_K_XL.gguf?download=true",
            SupportedModel::Llama3p2_3bQ4 => "https://pub-8987485129c64debb63bff7f35a2e5fd.r2.dev/v0/lmstudio-community/Llama-3.2-3B-Instruct-GGUF/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        }
    }

    pub fn model_size(&self) -> u64 {
        match self {
            SupportedModel::Qwen3_8b_Thinking => 5135722304, // 5GB estimate for Qwen3-8B-UD-Q4_K_XL
            SupportedModel::Llama3p2_3bQ4 => 2019377440,
        }
    }

    pub fn model_name(&self) -> &str {
        match self {
            SupportedModel::Qwen3_8b_Thinking => "qwen",
            SupportedModel::Llama3p2_3bQ4 => "llama",
        }
    }
}
