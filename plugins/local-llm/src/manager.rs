use std::sync::Arc;
use tokio::sync::{watch, Mutex};

#[derive(Clone)]
pub struct ModelManager {
    model_path: std::path::PathBuf,
    model: Arc<Mutex<Option<Arc<hypr_llama::Llama>>>>,
    last_activity: Arc<Mutex<Option<tokio::time::Instant>>>,
    _drop_guard: Arc<DropGuard>,
    current_model: Arc<Mutex<Option<crate::SupportedModel>>>,
}

struct DropGuard {
    shutdown_tx: watch::Sender<()>,
}

impl Drop for DropGuard {
    fn drop(&mut self) {
        let _ = self.shutdown_tx.send(());
    }
}

impl ModelManager {
    pub fn new(model_path: impl Into<std::path::PathBuf>) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(());

        let manager = Self {
            model_path: model_path.into(),
            model: Arc::new(tokio::sync::Mutex::new(None)),
            last_activity: Arc::new(tokio::sync::Mutex::new(None)),
            _drop_guard: Arc::new(DropGuard { shutdown_tx }),
            current_model: Arc::new(tokio::sync::Mutex::new(None)),
        };

        manager.monitor(shutdown_rx);
        manager
    }

    pub fn get_current_model(&self) -> Option<crate::SupportedModel> {
        match self.current_model.try_lock() {
            Ok(model) => {
                let result = model.clone();
                if let Some(model_type) = &result {
                    tracing::info!("Current model is {}", model_type.model_name());
                }
                result
            },
            Err(_) => {
                tracing::warn!("Could not get lock on current_model");
                None
            },
        }
    }

    pub async fn set_current_model(&self, model: crate::SupportedModel) {
        tracing::info!("Setting current model to: {}", model.model_name());
        {
            let mut guard = self.current_model.lock().await;
            *guard = Some(model.clone());
        }
        
        // Force model reload on next get_model
        {
            let mut model_guard = self.model.lock().await;
            if model_guard.is_some() {
                tracing::info!("Unloading existing model to force reload");
            }
            *model_guard = None;
        }
        
        // Update activity timestamp to prevent immediate unloading
        self.update_activity().await;
        
        tracing::info!("Current model successfully set to {}", model.model_name());
    }

    pub async fn update_activity(&self) {
        let now = tokio::time::Instant::now();
        let mut guard = self.last_activity.lock().await;
        
        // Only log if it's been a while since the last update
        if let Some(last) = *guard {
            if last.elapsed() > std::time::Duration::from_secs(10) {
                tracing::debug!("Updating model activity timestamp after {}s", last.elapsed().as_secs());
            }
        } else {
            tracing::debug!("Setting initial model activity timestamp");
        }
        
        *guard = Some(now);
    }

    pub async fn get_model(&self) -> Result<std::sync::Arc<hypr_llama::Llama>, crate::Error> {
        self.update_activity().await;

        let mut guard = self.model.lock().await;

        if let Some(model) = guard.as_ref() {
            tracing::debug!("Returning existing model instance");
            return Ok(model.clone());
        }
        
        // Get current model if available
        let current_model = {
            let model_guard = self.current_model.lock().await;
            model_guard.clone()
        };
        
        // Log detailed information about the model file
        if let Ok(metadata) = std::fs::metadata(&self.model_path) {
            tracing::info!("Model file stats: exists={}, size={}, is_file={}", 
                metadata.is_file(), metadata.len(), metadata.is_file());
        } else {
            tracing::warn!("Cannot read model file metadata at {:?}", self.model_path);
        }
        
        if let Some(model_type) = &current_model {
            tracing::info!("Creating new model instance for {} at path {:?}", 
                model_type.model_name(), self.model_path);
        } else {
            tracing::info!("Creating new model instance with default settings at path {:?}", 
                self.model_path);
        }
        
        // Try to validate GGUF format
        if let Ok(mut file) = std::fs::File::open(&self.model_path) {
            let mut magic = [0u8; 4];
            if let Ok(_) = std::io::Read::read_exact(&mut file, &mut magic) {
                tracing::info!("Model file magic bytes: {:02X?}", magic);
                if magic == [0x47, 0x47, 0x55, 0x46] { // "GGUF"
                    tracing::info!("Model file has valid GGUF magic number");
                } else {
                    tracing::warn!("Model file does NOT have valid GGUF magic number");
                }
            }
        }
        
        // Try to load the model
        tracing::info!("Attempting to load model with llama-cpp");
        match hypr_llama::Llama::new(&self.model_path) {
            Ok(llama) => {
                let model = Arc::new(llama);
                tracing::info!("Model loaded successfully");
                *guard = Some(model.clone());
                Ok(model)
            },
            Err(e) => {
                tracing::error!("Failed to load model: {} (error type: {})", e, std::any::type_name_of_val(&e));
                
                // Try to get more information about what might have gone wrong
                if e.to_string().contains("null result") {
                    tracing::error!("Null result error typically indicates:"); 
                    tracing::error!("1. The model file is corrupt or incomplete");
                    tracing::error!("2. The model format is not compatible with this version of llama-cpp");
                    tracing::error!("3. There might be a permission issue accessing the file");
                    
                    // Get first 100 bytes of file to check if it's valid
                    if let Ok(mut file) = std::fs::File::open(&self.model_path) {
                        let mut buffer = [0u8; 100];
                        if let Ok(bytes_read) = std::io::Read::read(&mut file, &mut buffer) {
                            tracing::error!("First {} bytes of model file: {:02X?}", 
                                bytes_read, &buffer[..bytes_read.min(16)]);
                        }
                    }
                }
                
                Err(crate::Error::HyprLlamaError(e))
            }
        }
    }

    fn monitor(&self, shutdown_rx: watch::Receiver<()>) {
        let activity_check_interval = std::time::Duration::from_secs(3);
        let inactivity_threshold = std::time::Duration::from_secs(150);

        let model = self.model.clone();
        let last_activity = self.last_activity.clone();
        let model_path = self.model_path.clone();
        let current_model = self.current_model.clone();

        let _handle = tokio::spawn(async move {
        let mut shutdown_rx = shutdown_rx;
        let mut interval = tokio::time::interval(activity_check_interval);

        tracing::info!("Starting model monitor for path: {:?}", model_path);
        interval.tick().await;

            loop {
                tokio::select! {
                    _ = shutdown_rx.changed() => {
                        tracing::info!("Shutting down model monitor");
                        break;
                    },
                    _ = interval.tick() => {
                        let should_unload = match *last_activity.lock().await {
                            Some(last_time) if last_time.elapsed() > inactivity_threshold => {
                                let has_model = model.lock().await.is_some();
                                if has_model {
                                    // Get model name for logging
                                    let model_name = if let Ok(guard) = current_model.try_lock() {
                                        if let Some(ref model_type) = *guard {
                                            model_type.model_name().to_string()
                                        } else {
                                            "unknown".to_string()
                                        }
                                    } else {
                                        "unknown".to_string()
                                    };
                                    
                                    tracing::info!("Unloading {} model due to inactivity ({}s)", 
                                        model_name, last_time.elapsed().as_secs());
                                }
                                has_model
                            },
                            Some(last_time) => {
                                tracing::debug!("Model active: last activity was {}s ago", 
                                    last_time.elapsed().as_secs());
                                false
                            },
                            None => {
                                tracing::debug!("No activity recorded yet");
                                false
                            }
                        };

                        if should_unload {
                            tracing::info!("Unloading model to free memory");
                            *model.lock().await = None;
                        }
                    }
                }
            }
        });
    }
}
