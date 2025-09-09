use crate::{LocalLlmPluginExt, SupportedModel, SUPPORTED_MODELS};

use ollama_rs::Ollama;
use tauri::ipc::Channel;

#[tauri::command]
#[specta::specta]
pub async fn is_server_running<R: tauri::Runtime>(app: tauri::AppHandle<R>) -> bool {
    app.is_server_running().await
}

#[tauri::command]
#[specta::specta]
pub async fn is_model_downloaded<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
) -> Result<bool, String> {
    tracing::info!("is_model_downloaded called");
    let result = app.is_model_downloaded().await.map_err(|e| {
        tracing::error!("Error checking if model is downloaded: {}", e);
        e.to_string()
    });
    if let Ok(downloaded) = result {
        tracing::info!("Model downloaded: {}", downloaded);
    }
    result
}

#[tauri::command]
#[specta::specta]
pub async fn is_model_downloading<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
) -> Result<bool, String> {
    Ok(app.is_model_downloading().await)
}

#[tauri::command]
#[specta::specta]
pub async fn download_model<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    channel: Channel<i8>,
) -> Result<(), String> {
    tracing::info!("download_model called");
    match app.download_model(channel).await {
        Ok(_) => {
            tracing::info!("Model download initiated successfully");
            Ok(())
        }
        Err(e) => {
            tracing::error!("Error initiating model download: {}", e);
            Err(e.to_string())
        }
    }
}

#[tauri::command]
#[specta::specta]
pub async fn start_server<R: tauri::Runtime>(app: tauri::AppHandle<R>) -> Result<String, String> {
    app.start_server().await.map_err(|e| e.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn stop_server<R: tauri::Runtime>(app: tauri::AppHandle<R>) -> Result<(), String> {
    app.stop_server().await.map_err(|e| e.to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn list_ollama_models<R: tauri::Runtime>(
    _app: tauri::AppHandle<R>,
) -> Result<Vec<String>, String> {
    let ollama = Ollama::default();
    let models = ollama
        .list_local_models()
        .await
        .map_err(|e| e.to_string())?;

    Ok(models.into_iter().map(|m| m.name).collect::<Vec<_>>())
}

#[tauri::command]
#[specta::specta]
pub async fn list_supported_models<R: tauri::Runtime>(
    _app: tauri::AppHandle<R>,
) -> Result<Vec<SupportedModel>, String> {
    tracing::info!("list_supported_models called");
    let models = SUPPORTED_MODELS.to_vec();
    tracing::info!("Returning {} supported models", models.len());
    Ok(models)
}

#[tauri::command]
#[specta::specta]
pub async fn get_current_model<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
) -> Result<SupportedModel, String> {
    tracing::info!("get_current_model called");
    let result = app.current_model().await.map_err(|e| {
        tracing::error!("Error getting current model: {}", e);
        e.to_string()
    });
    if let Ok(_model) = &result {
        tracing::info!("Current model is set");
    }
    result
}

#[tauri::command]
#[specta::specta]
pub async fn set_current_model<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    model: SupportedModel,
) -> Result<(), String> {
    tracing::info!("set_current_model called with model");
    let store = app.local_llm_store();
    match store.set(crate::StoreKey::Model, Some(model)) {
        Ok(_) => {
            tracing::info!("Successfully set current model");
            Ok(())
        }
        Err(e) => {
            tracing::error!("Error setting current model: {}", e);
            Err(e.to_string())
        }
    }
}
