import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";

import { commands as localLlmCommands, SupportedModel as LlmSupportedModel } from "@hypr/plugin-local-llm";
import { commands as localSttCommands, SupportedModel as SttSupportedModel } from "@hypr/plugin-local-stt";
import { commands as windowsCommands } from "@hypr/plugin-windows";
import { Button } from "@hypr/ui/components/ui/button";
import { sonnerToast, toast } from "@hypr/ui/components/ui/toast";
import { showLlmModelDownloadToast, showSttModelDownloadToast } from "./shared";

export default function ModelDownloadNotification() {
  const currentSttModel = useQuery({
    queryKey: ["current-stt-model"],
    queryFn: () => localSttCommands.getCurrentModel() as Promise<SttSupportedModel>,
  });

  const currentLlmModel = useQuery({
    queryKey: ["current-llm-model"],
    queryFn: () => localLlmCommands.getCurrentModel() as Promise<LlmSupportedModel>,
  });

  const checkForModelDownload = useQuery({
    enabled: !!currentSttModel.data && !!currentLlmModel.data,
    queryKey: ["check-model-downloaded"],
    queryFn: async () => {
      const [stt, llm] = await Promise.all([
        localSttCommands.isModelDownloaded(currentSttModel.data!),
        localLlmCommands.isModelDownloaded(),
      ]);

      return {
        currentSttModel,
        sttModelDownloaded: stt,
        llmModelDownloaded: llm,
      };
    },
    refetchInterval: 5000,
  });

  const sttModelDownloading = useQuery({
    enabled: !checkForModelDownload.data?.sttModelDownloaded,
    queryKey: ["stt-model-downloading"],
    queryFn: async () => {
      return localSttCommands.isModelDownloading(currentSttModel.data!);
    },
    refetchInterval: 3000,
  });

  const llmModelDownloading = useQuery({
    enabled: !checkForModelDownload.data?.llmModelDownloaded,
    queryKey: ["llm-model-downloading"],
    queryFn: async () => {
      return localLlmCommands.isModelDownloading();
    },
    refetchInterval: 3000,
  });

  useEffect(() => {
    if (!checkForModelDownload.data) {
      return;
    }

    if (checkForModelDownload.data?.sttModelDownloaded && checkForModelDownload.data?.llmModelDownloaded) {
      return;
    }

    if (sttModelDownloading.data || llmModelDownloading.data) {
      return;
    }

    toast({
      id: "model-download-needed",
      title: "Model Download Needed",
      content: (
        <div className="space-y-2">
          <p>Local models are required for offline functionality.</p>
          <div className="flex flex-col gap-2">
            {!checkForModelDownload.data?.sttModelDownloaded && !sttModelDownloading.data && currentSttModel.data && (
              <Button 
                size="sm" 
                onClick={() => showSttModelDownloadToast(currentSttModel.data!)}
              >
                Download Speech-to-Text Model
              </Button>
            )}
            {!checkForModelDownload.data?.llmModelDownloaded && !llmModelDownloading.data && currentLlmModel.data && (
              <Button 
                size="sm" 
                onClick={() => showLlmModelDownloadToast(currentLlmModel.data!)}
              >
                Download Language Model
              </Button>
            )}
            {(!currentLlmModel.data || !currentSttModel.data) && (
              <Button 
                size="sm" 
                onClick={() => {
                  windowsCommands.windowShow({ type: "settings" });
                  sonnerToast.dismiss("model-download-needed");
                }}
              >
                Open Settings
              </Button>
            )}
          </div>
        </div>
      ),
      dismissible: true,
    });
  }, [checkForModelDownload.data, sttModelDownloading.data, llmModelDownloading.data]);

  return null;
}
