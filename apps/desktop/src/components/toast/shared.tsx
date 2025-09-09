import { Channel } from "@tauri-apps/api/core";
import { useEffect, useState } from "react";

import { commands as localLlmCommands, SupportedModel as LlmSupportedModel } from "@hypr/plugin-local-llm";
import { commands as localSttCommands, SupportedModel as SttSupportedModel } from "@hypr/plugin-local-stt";
import { commands as windowsCommands } from "@hypr/plugin-windows";
import { Button } from "@hypr/ui/components/ui/button";
import { Progress } from "@hypr/ui/components/ui/progress";
import { sonnerToast, toast } from "@hypr/ui/components/ui/toast";

export const DownloadProgress = ({
  channel,
  onComplete,
}: {
  channel: Channel<number>;
  onComplete?: () => void;
}) => {
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(false);

  useEffect(() => {
    let mounted = true;
    let completionTimer: NodeJS.Timeout | null = null;
    let autoCompleteTimer: NodeJS.Timeout | null = null;
    
    const handleMessage = (v: number) => {
      if (!mounted) return;
      
      if (v < 0) {
        setError(true);
        return;
      }

      if (v > progress) {
        setProgress(v);
      }

      if (v >= 100 && onComplete && mounted) {
        // Slight delay to ensure UI updates before completion
        completionTimer = setTimeout(() => {
          if (mounted && onComplete) {
            onComplete();
          }
        }, 800);
      }
    };
    
    channel.onmessage = handleMessage;
    
    // Auto-complete after 10 seconds if the progress is high (>95%)
    autoCompleteTimer = setTimeout(() => {
      if (mounted && progress > 95 && onComplete) {
        console.log("Auto-completing download after timeout");
        onComplete();
      }
    }, 10000);
    
    return () => {
      // Cleanup when component unmounts
      mounted = false;
      if (completionTimer) {
        clearTimeout(completionTimer);
      }
      if (autoCompleteTimer) {
        clearTimeout(autoCompleteTimer);
      }
      channel.onmessage = null;
    };
  }, [channel, onComplete, progress]);

  if (error) {
    return (
      <div className="w-full">
        <div className="text-destructive font-medium">Download failed. Please try again.</div>
      </div>
    );
  }

  return (
    <div className="w-full space-y-2">
      <Progress value={progress} className="h-2" />
      <div className="text-xs text-right">{Math.round(progress)}%</div>
    </div>
  );
};

export function showSttModelDownloadToast(model: SttSupportedModel, onComplete?: () => void) {
  const sttChannel = new Channel();
  const id = `stt-model-download-${model}`;
  
  // Prevent duplicate toasts
  sonnerToast.dismiss(id);
  
  // Initiate download and handle errors
  localSttCommands.downloadModel(model, sttChannel)
    .catch(() => {
      sonnerToast.dismiss(id);
      toast({
        id: `${id}-error`,
        title: "Download Error",
        content: "Failed to start STT model download. Please try again.",
        dismissible: true,
      });
    });

  toast(
    {
      id,
      title: "Speech-to-Text Model",
      content: (
        <div className="space-y-1">
          <div>Downloading speech-to-text model: {model}...</div>
          <DownloadProgress
            channel={sttChannel}
            onComplete={() => {
              sonnerToast.dismiss(id);
              localSttCommands.startServer().catch(() => {
                toast({
                  id: `${id}-server-error`,
                  title: "Server Error",
                  content: "Failed to start the STT server. The model may not have downloaded properly.",
                  dismissible: true,
                });
              });
              if (onComplete) {
                onComplete();
              }
            }}
          />
        </div>
      ),
      dismissible: true,
    },
  );
}

export function showLlmModelDownloadToast(model: LlmSupportedModel, onComplete?: () => void) {
  const llmChannel = new Channel();
  const id = `llm-model-download-${model}`;
  const modelName = getDisplayModelName(model);
  
  // Log the model being downloaded
  console.log(`Downloading model: ${model} (Display name: ${modelName})`);

  // Prevent duplicate toasts
  sonnerToast.dismiss(id);
  
  // Create a timeout to force-dismiss the toast after 30 seconds
  const forceDismissTimeout = setTimeout(() => {
    sonnerToast.dismiss(id);
    
    // Try to start the server anyway
    localLlmCommands.startServer().catch(e => console.error("Error starting server:", e));
    
    if (onComplete) {
      onComplete();
    }
  }, 30000);
  
  // Initiate download and handle errors
  localLlmCommands.downloadModel(llmChannel)
    .catch(() => {
      clearTimeout(forceDismissTimeout);
      sonnerToast.dismiss(id);
      toast({
        id: `${id}-error`,
        title: "Download Error",
        content: "Failed to start model download. Please try again.",
        dismissible: true,
      });
    });

  toast(
    {
      id,
      title: "Downloading Language Model",
      content: (
        <div className="space-y-1">
          <div>Downloading {modelName} model...</div>
          <DownloadProgress
            channel={llmChannel}
            onComplete={() => {
              clearTimeout(forceDismissTimeout);
              sonnerToast.dismiss(id);
              
              // Start the server with the new model
              localLlmCommands.startServer()
                .catch(() => {
                  toast({
                    id: `${id}-server-error`,
                    title: "Server Error",
                    content: "Failed to start the model server. The model may not have downloaded properly.",
                    dismissible: true,
                  });
                });
                
              if (onComplete) {
                onComplete();
              }
            }}
          />
        </div>
      ),
      dismissible: true,
      onDismiss: () => {
        clearTimeout(forceDismissTimeout);
      }
    },
  );
}

function getDisplayModelName(model: LlmSupportedModel): string {
  const displayNames: Record<string, string> = {
    "Qwen3_8b_Thinking": "Qwen3 8B (Thinking)",
    "Llama3p2_3bQ4": "Llama 3.2 3B"
  };
  
  // Ensure we get a display name even if the model isn't in our mapping
  try {
    return displayNames[model] || model.toString().replace(/_/g, " ");
  } catch (e) {
    console.error("Error getting display name for model:", model, e);
    return model ? model.toString() : "Unknown Model";
  }
}

export function enhanceFailedToast() {
  const id = "no-llm-connection";

  const handleClick = () => {
    windowsCommands.windowShow({ type: "settings" });
    sonnerToast.dismiss(id);
  };

  toast({
    id,
    title: "Failed to enhance meeting notes",
    content: (
      <div className="space-y-1">
        <div>Go to AI settings to check the status.</div>
        <Button variant="default" onClick={handleClick}>
          Open Settings
        </Button>
      </div>
    ),
    dismissible: true,
  });
}
