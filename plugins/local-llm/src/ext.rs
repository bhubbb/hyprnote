use std::future::Future;

use hypr_file::{download_file_with_callback, DownloadProgress};
use tauri::{ipc::Channel, Manager, Runtime};
use tauri_plugin_store2::StorePluginExt;

pub trait LocalLlmPluginExt<R: Runtime> {
    fn local_llm_store(&self) -> tauri_plugin_store2::ScopedStore<R, crate::StoreKey>;
    fn current_model(&self) -> impl Future<Output = Result<crate::SupportedModel, crate::Error>>;
    fn api_base(&self) -> impl Future<Output = Option<String>>;
    fn is_model_downloading(&self) -> impl Future<Output = bool>;
    fn is_model_downloaded(&self) -> impl Future<Output = Result<bool, crate::Error>>;
    fn is_server_running(&self) -> impl Future<Output = bool>;
    fn download_model(
        &self,
        channel: Channel<i8>,
    ) -> impl Future<Output = Result<(), crate::Error>>;
    fn start_server(&self) -> impl Future<Output = Result<String, crate::Error>>;
    fn stop_server(&self) -> impl Future<Output = Result<(), crate::Error>>;
}

impl<R: Runtime, T: Manager<R>> LocalLlmPluginExt<R> for T {
    fn local_llm_store(&self) -> tauri_plugin_store2::ScopedStore<R, crate::StoreKey> {
        self.scoped_store(crate::PLUGIN_NAME).unwrap()
    }

    async fn current_model(&self) -> Result<crate::SupportedModel, crate::Error> {
        let store = self.local_llm_store();

        let stored = store
            .get::<Option<crate::SupportedModel>>(crate::StoreKey::Model)?
            .flatten();
        
        let model = stored.unwrap_or(crate::SupportedModel::Qwen3_8b_Thinking);
        tracing::debug!("Current model: {:?}, architecture: {}, size: {} bytes", 
                       model, model.model_name(), model.model_size());
        Ok(model)
    }

    #[tracing::instrument(skip_all)]
    async fn api_base(&self) -> Option<String> {
        let state = self.state::<crate::SharedState>();
        let s = state.lock().await;
        s.api_base.clone()
    }

    #[tracing::instrument(skip_all)]
    async fn is_model_downloading(&self) -> bool {
        let state = self.state::<crate::SharedState>();
        let s = state.lock().await;
        s.download_task.is_some()
    }

    #[tracing::instrument(skip_all)]
    async fn is_model_downloaded(&self) -> Result<bool, crate::Error> {
        let model = self.current_model().await?;

        let data_dir = self.path().app_data_dir().unwrap();
        let path = model.model_path(data_dir);

        tracing::info!("Checking if model is downloaded: {:?}", path);
        tracing::info!("Current model: {} (URL: {})", 
                      model.model_name(), model.model_url());

        if !path.exists() {
            tracing::info!("Model file not found at {:?}", path);
            return Ok(false);
        }

        let size = hypr_file::file_size(&path)?;
        let expected_size = model.model_size();
        let size_match = size == expected_size;
        
        tracing::info!("Model file stats: exists=true, size={}, expected={}, match={}", 
                     size, expected_size, size_match);
        
        // If size doesn't match, log additional diagnostics
        if !size_match {
            tracing::warn!("Model file size mismatch - may be incomplete download");
            tracing::warn!("  - File size: {} bytes", size);
            tracing::warn!("  - Expected:  {} bytes", expected_size);
            tracing::warn!("  - Difference: {} bytes", 
                          if size > expected_size { size - expected_size } else { expected_size - size });
            
            if size < 1024 * 1024 {
                // Very small file - likely just a placeholder or error
                tracing::warn!("File size < 1MB, likely not a valid model");
            }
        }
        
        // Basic content validation for debugging
        if let Ok(mut file) = std::fs::File::open(&path) {
            let mut header = [0u8; 16];
            if let Ok(bytes_read) = std::io::Read::read(&mut file, &mut header) {
                tracing::info!("First {} bytes of model file: {:02X?}", bytes_read, &header[..bytes_read]);
            }
        }
        
        Ok(size_match)
    }

    #[tracing::instrument(skip_all)]
    async fn is_server_running(&self) -> bool {
        let state = self.state::<crate::SharedState>();
        let s = state.lock().await;
        s.server.is_some()
    }

    #[tracing::instrument(skip_all)]
    async fn download_model(&self, channel: Channel<i8>) -> Result<(), crate::Error> {
        let model = self.current_model().await?;
        let data_dir = self.path().app_data_dir().unwrap();

        let path = model.model_path(data_dir);
        let url = model.model_url().to_string();
        
        tracing::info!("Starting download for model {} at URL: {}", model.model_name(), url);
        tracing::info!("Download destination: {:?}", path);
        tracing::info!("Expected model size: {} bytes", model.model_size());
        
        // Check if we're overwriting an existing file
        if path.exists() {
            if let Ok(metadata) = std::fs::metadata(&path) {
                tracing::info!("Existing file found: {} bytes", metadata.len());
                if metadata.len() == model.model_size() {
                    tracing::info!("Existing file has correct size - may be already downloaded");
                } else {
                    tracing::warn!("Existing file has incorrect size - will be overwritten");
                }
            }
        }

        // Clone path for use in the callback to avoid borrow issues
        let path_display = path.display().to_string();
        let path_for_validation = path.clone();
        
        let task = tokio::spawn(async move {
            let callback = |progress: DownloadProgress| match progress {
                DownloadProgress::Started => {
                    tracing::info!("Download started for {} (expected size: {} bytes)", 
                                  path_display, model.model_size());
                    let _ = channel.send(0);
                }
                DownloadProgress::Progress(downloaded, total_size) => {
                    let percent = (downloaded as f64 / total_size as f64) * 100.0;
                    let _ = channel.send(percent as i8);
                    
                    if downloaded % (50 * 1024 * 1024) < 1024 * 1024 {  // Log every ~50MB
                        tracing::info!("Download progress: {:.1}% ({} / {} bytes of {})", 
                                     percent, downloaded, total_size, model.model_size());
                        
                        // Check if reported total size matches expected size
                        if total_size != model.model_size() {
                            tracing::warn!("Server reported size ({}) differs from expected size ({})",
                                          total_size, model.model_size());
                        }
                    }
                }
                DownloadProgress::Finished => {
                    tracing::info!("Download finished, validating file...");
                    
                    // Validate downloaded file
                    if let Ok(metadata) = std::fs::metadata(&path_for_validation) {
                        tracing::info!("Downloaded file size: {} bytes", metadata.len());
                        if metadata.len() == model.model_size() {
                            tracing::info!("File size matches expected size");
                        } else {
                            tracing::error!("Downloaded file size ({}) does not match expected size ({})",
                                           metadata.len(), model.model_size());
                        }
                    } else {
                        tracing::error!("Could not read metadata for downloaded file");
                    }
                    
                    let _ = channel.send(100);
                }
            };

            if let Err(e) = download_file_with_callback(url, path, callback).await {
                tracing::error!("model_download_error: {} (type: {})", 
                               e, std::any::type_name_of_val(&e));
                let _ = channel.send(-1);
            }
        });

        {
            let state = self.state::<crate::SharedState>();
            let mut s = state.lock().await;

            if let Some(task) = s.download_task.take() {
                tracing::info!("Cancelling previous download");
                task.abort();
            }
            s.download_task = Some(task);
        }

        Ok(())
    }

    #[tracing::instrument(skip_all)]
    async fn start_server(&self) -> Result<String, crate::Error> {
        let state = self.state::<crate::SharedState>();

        // Check if the model is downloaded
        if !self.is_model_downloaded().await? {
            tracing::error!("Cannot start server - model not downloaded");
            return Err(crate::Error::ModelNotDownloaded);
        }

        // Get the current model before creating the model manager
        let current_model = self.current_model().await?;
        let data_dir = self.path().app_data_dir().unwrap();
        let model_path = current_model.model_path(&data_dir);
        
        tracing::info!("Starting server with model: {}", current_model.model_name());
        tracing::info!("Model path: {:?}, file exists: {}", 
                      model_path, model_path.exists());
        
        // Check file details
        if let Ok(metadata) = std::fs::metadata(&model_path) {
            tracing::info!("Model file size: {} bytes, expected: {} bytes", 
                          metadata.len(), current_model.model_size());
        } else {
            tracing::warn!("Could not read model file metadata");
        }
        
        // Try to check GGUF magic
        if let Ok(mut file) = std::fs::File::open(&model_path) {
            let mut magic = [0u8; 4];
            if let Ok(_) = std::io::Read::read_exact(&mut file, &mut magic) {
                tracing::info!("Model file magic: {:02X?}", magic);
                if magic == [0x47, 0x47, 0x55, 0x46] { // "GGUF"
                    tracing::info!("Valid GGUF magic found");
                } else {
                    tracing::warn!("Invalid magic bytes - file may be corrupt");
                }
            } else {
                tracing::warn!("Could not read file magic");
            }
        }
        
        let model_manager = {
            let s = state.lock().await;
            tracing::info!("Creating model manager with path: {:?}", s.model_path);
            crate::ModelManager::new(s.model_path.clone())
        };
        
        // Set the current model in the model manager
        model_manager.set_current_model(current_model).await;

        // Attempt to start the server
        let server_result = crate::server::run_server(model_manager).await;
        
        match server_result {
            Ok(server) => {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                let api_base = format!("http://{}", &server.addr);
                tracing::info!("Server started successfully at: {}", api_base);
                
                {
                    let mut s = state.lock().await;
                    s.api_base = Some(api_base.clone());
                    s.server = Some(server);
                }
                
                Ok(api_base)
            },
            Err(e) => {
                tracing::error!("Failed to start server: {}", e);
                tracing::error!("Error type: {}", std::any::type_name_of_val(&e));
                
                if e.to_string().contains("null result") {
                    tracing::error!("Likely causes for null result error:");
                    tracing::error!("1. Corrupt model file");
                    tracing::error!("2. Incompatible model format");
                    tracing::error!("3. File system permission issues");
                    tracing::error!("4. Model architecture mismatch");
                }
                
                Err(e)
            }
        }
    }

    #[tracing::instrument(skip_all)]
    async fn stop_server(&self) -> Result<(), crate::Error> {
        let state = self.state::<crate::SharedState>();
        let mut s = state.lock().await;

        if let Some(server) = s.server.take() {
            let _ = server.shutdown.send(());
        }
        Ok(())
    }
}
