import { useEffect, useState } from "react";
import type { ChangeEvent } from "react";
import "./App.css";

type HealthResponse = {
  status: string;
};

type AudioMetadata = {
  duration: number;
  sample_rate: number;
  channels: number;
  frames: number;
  format: string;
  subtype: string;
};

type UploadResponse = {
  message: string;
  original_name: string;
  stored_name: string;
  processed_name: string;
  size: number;
  content_type: string;
  metadata: AudioMetadata;
};

function App() {
  const [backendStatus, setBackendStatus] = useState("Checking backend...");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadMessage, setUploadMessage] = useState("");
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch(
          `${import.meta.env.VITE_API_BASE_URL}/health`
        );

        const data: HealthResponse = await response.json();

        if (data.status === "ok") {
          setBackendStatus("Backend connected");
        } else {
          setBackendStatus("Backend not connected");
        }
      } catch {
        setBackendStatus("Backend not connected");
      }
    };

    checkBackend();
  }, []);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];

    if (!file) {
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
    setUploadMessage("");
    setUploadResult(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadMessage("Select an audio file first");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    setIsUploading(true);
    setUploadMessage("");
    setUploadResult(null);

    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_BASE_URL}/upload-audio`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        setUploadMessage(data.detail || "Upload failed");
        return;
      }

      const result: UploadResponse = data;

      setUploadResult(result);
      setUploadMessage(
        `${result.original_name} uploaded successfully`
      );
    } catch {
      setUploadMessage("Unable to upload audio");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <main className="app">
      <section className="hero">
        <p className="eyebrow">Audio-Driven Motion Generation</p>

        <h1>Digital Human Animation</h1>

        <p className="description">
          Generate synchronized digital human animation from speech and audio.
        </p>

        <div className="status">
          <span>System Status</span>
          <strong>{backendStatus}</strong>
        </div>

        <div className="upload-card">
          <h2>Upload Audio</h2>

          <p>Select a speech recording to begin the animation process.</p>

          <input
            type="file"
            accept=".mp3,.wav,.m4a,.aac,audio/*"
            onChange={handleFileChange}
          />

          {selectedFile && (
            <div className="file-info">
              <span>{selectedFile.name}</span>
              <span>{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span>
            </div>
          )}

          <button
            type="button"
            onClick={handleUpload}
            disabled={isUploading}
          >
            {isUploading ? "Uploading..." : "Upload Audio"}
          </button>

          {uploadMessage && (
            <p className="upload-message">{uploadMessage}</p>
          )}

          {uploadResult && (
            <div className="metadata">
              <div>
                <span>Duration</span>
                <strong>{uploadResult.metadata.duration} sec</strong>
              </div>

              <div>
                <span>Sample Rate</span>
                <strong>{uploadResult.metadata.sample_rate} Hz</strong>
              </div>

              <div>
                <span>Channels</span>
                <strong>{uploadResult.metadata.channels}</strong>
              </div>

              <div>
                <span>Format</span>
                <strong>{uploadResult.metadata.format}</strong>
              </div>
            </div>
          )}
        </div>
      </section>
    </main>
  );
}

export default App;