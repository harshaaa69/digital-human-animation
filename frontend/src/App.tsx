import { Suspense, useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { WebGLRenderer } from "three";
import AvatarModel from "./AvatarModel";
import "./App.css";
import "./ExportVideo.css";
import "./MotionCompare.css";
import "./AudioUpload.css";
import "./VoiceSpeed.css";

type MotionMode = "baseline" | "viseme" | "smoothed";
type ProcessingStage = "idle" | "uploading" | "processing" | "loading" | "ready" | "error";
type ExportStage = "idle" | "recording" | "processing" | "completed" | "error";
type VoiceSpeed = 1 | 0.95 | 0.9 | 0.85;

type MotionResponse = {
  motion_name: string;
  fps: number;
  frames: number;
  parameters: string[];
  data: number[][];
};

type UploadResponse = {
  processed_name: string;
  original_name: string;
};

type ProcessingResponse = {
  audio_name: string;
  audio_url: string;
  duration: number;
  voice_speed: number;
  motion_urls: Record<MotionMode, string>;
  transcription: string;
  viseme_segments: number;
};

type ExportResponse = {
  message: string;
  video_name: string;
  download_url: string;
  size_bytes: number;
};

type AnimationSession = {
  audioName: string;
  audioUrl: string;
  filename: string;
  voiceSpeed: number;
  transcription: string;
  visemeSegments: number;
  motions: Record<MotionMode, MotionResponse>;
};

const API = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const MODES: { id: MotionMode; label: string }[] = [
  { id: "baseline", label: "Baseline" },
  { id: "viseme", label: "Viseme-Enhanced" },
  { id: "smoothed", label: "Smoothed Viseme" }
];
const SPEED_OPTIONS: { value: VoiceSpeed; label: string }[] = [
  { value: 1, label: "100% · Original" },
  { value: 0.95, label: "95% · Gentle" },
  { value: 0.9, label: "90% · Relaxed" },
  { value: 0.85, label: "85% · Slow" }
];
const ACCEPTED_EXTENSIONS = [".wav", ".mp3", ".m4a", ".aac"];

function formatTime(value: number): string {
  const seconds = Number.isFinite(value) ? Math.max(0, value) : 0;
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, "0")}:${(seconds % 60).toFixed(2).padStart(5, "0")}`;
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Unexpected error";
}

async function readResponse<T>(response: Response): Promise<T> {
  const result: unknown = await response.json();
  if (!response.ok) {
    if (result && typeof result === "object" && "detail" in result && typeof result.detail === "string") {
      throw new Error(result.detail);
    }
    throw new Error(`Request failed (${response.status})`);
  }
  return result as T;
}

function generateWaveform(samples: Float32Array, bars = 110): number[] {
  const levels: number[] = [];
  for (let i = 0; i < bars; i++) {
    const start = Math.floor(i * samples.length / bars);
    const end = Math.floor((i + 1) * samples.length / bars);
    let power = 0;
    for (let j = start; j < end; j++) power += samples[j] * samples[j];
    levels.push(Math.sqrt(power / Math.max(1, end - start)));
  }
  const max = Math.max(0.0001, ...levels);
  return levels.map((level) => Math.max(0.06, Math.min(1, level / max)));
}

function getStatistics(motion?: MotionResponse) {
  if (!motion || motion.frames < 2) return null;
  const values = motion.data.map((frame) => frame[0]);
  const changes = values.slice(1).map((value, index) => Math.abs(value - values[index]));
  return {
    mean: values.reduce((a, b) => a + b, 0) / values.length,
    meanChange: changes.reduce((a, b) => a + b, 0) / changes.length,
    maxChange: Math.max(...changes),
    maxOpening: Math.max(...values)
  };
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [voiceSpeed, setVoiceSpeed] = useState<VoiceSpeed>(0.9);
  const [session, setSession] = useState<AnimationSession | null>(null);
  const [stage, setStage] = useState<ProcessingStage>("idle");
  const [processingError, setProcessingError] = useState("");
  const [mode, setMode] = useState<MotionMode>("baseline");
  const [frameIndex, setFrameIndex] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [waveform, setWaveform] = useState<number[]>([]);
  const [exportStage, setExportStage] = useState<ExportStage>("idle");
  const [exportProgress, setExportProgress] = useState(0);
  const [exportError, setExportError] = useState("");
  const [exportResult, setExportResult] = useState<ExportResponse | null>(null);

  const audioRef = useRef<HTMLAudioElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const rendererRef = useRef<WebGLRenderer | null>(null);
  const exportingRef = useRef(false);
  const processingRef = useRef(false);

  const motion = session?.motions[mode];
  const duration = motion ? motion.frames / motion.fps : 0;
  const isProcessing = stage === "uploading" || stage === "processing" || stage === "loading";
  const isExporting = exportStage === "recording" || exportStage === "processing";
  const frame = motion?.data[Math.min(frameIndex, motion.frames - 1)] ?? [0, 0, 0, 0, 0, 0, 0];
  const statistics = getStatistics(motion);
  const progress = duration > 0 ? Math.min(100, currentTime / duration * 100) : 0;

  useEffect(() => {
    if (!session) {
      setWaveform([]);
      return;
    }
    const controller = new AbortController();
    let context: AudioContext | null = null;
    async function loadWaveform() {
      try {
        const response = await fetch(`${API}${session!.audioUrl}`, { signal: controller.signal });
        if (!response.ok) throw new Error("Cannot load waveform");
        const buffer = await response.arrayBuffer();
        if (controller.signal.aborted) return;
        context = new AudioContext();
        const decoded = await context.decodeAudioData(buffer);
        if (!controller.signal.aborted) setWaveform(generateWaveform(decoded.getChannelData(0)));
      } catch {
        if (!controller.signal.aborted) setWaveform([]);
      } finally {
        if (context) void context.close().catch(() => {});
      }
    }
    void loadWaveform();
    return () => controller.abort();
  }, [session]);

  function updatePosition(seconds: number, activeMotion: MotionResponse) {
    const maximum = activeMotion.frames / activeMotion.fps;
    const time = Math.max(0, Math.min(seconds, maximum));
    setCurrentTime(time);
    setFrameIndex(Math.min(activeMotion.frames - 1, Math.floor(time * activeMotion.fps)));
  }

  useEffect(() => {
    if (!isPlaying || !motion) return;
    let animationId = 0;
    const tick = () => {
      const audio = audioRef.current;
      if (!audio || audio.paused || audio.ended) return;
      updatePosition(audio.currentTime, motion);
      if (exportingRef.current) setExportProgress(Math.min(100, audio.currentTime / duration * 100));
      animationId = requestAnimationFrame(tick);
    };
    animationId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animationId);
  }, [isPlaying, motion, duration]);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    if (!file) return;
    const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(extension)) {
      setProcessingError("Unsupported audio format");
      setSelectedFile(null);
      return;
    }
    if (file.size === 0 || file.size > 25 * 1024 * 1024) {
      setProcessingError("Choose a non-empty audio file up to 25 MB");
      setSelectedFile(null);
      return;
    }
    setSelectedFile(file);
    setProcessingError("");
  }

  async function handleProcess() {
    if (!selectedFile || processingRef.current || exportingRef.current) return;
    processingRef.current = true;
    audioRef.current?.pause();
    setIsPlaying(false);
    setProcessingError("");
    setSession(null);
    setWaveform([]);
    setCurrentTime(0);
    setFrameIndex(0);
    setExportResult(null);
    setExportStage("idle");
    setExportError("");
    setMode("baseline");
    try {
      setStage("uploading");
      const uploadData = new FormData();
      uploadData.append("file", selectedFile);
      const uploadResponse = await fetch(`${API}/upload-audio`, { method: "POST", body: uploadData });
      const uploaded = await readResponse<UploadResponse>(uploadResponse);

      setStage("processing");
      const processingResponse = await fetch(
        `${API}/process-audio/${encodeURIComponent(uploaded.processed_name)}?speed=${voiceSpeed}`,
        { method: "POST" }
      );
      const processed = await readResponse<ProcessingResponse>(processingResponse);
      setStage("loading");

      const entries = await Promise.all(MODES.map(async (item) => {
        const response = await fetch(`${API}${processed.motion_urls[item.id]}`);
        const data = await readResponse<MotionResponse>(response);
        if (!Number.isFinite(data.fps) || data.fps <= 0 || data.frames <= 0 ||
            !Array.isArray(data.parameters) || data.parameters.length !== 7 ||
            !Array.isArray(data.data) || data.data.length !== data.frames ||
            data.data.some((row) => !Array.isArray(row) || row.length !== 7 || row.some((v) => !Number.isFinite(v)))) {
          throw new Error(`Invalid motion data for ${item.label}`);
        }
        return [item.id, data] as const;
      }));
      const motions = Object.fromEntries(entries) as Record<MotionMode, MotionResponse>;
      if (MODES.some((item) => motions[item.id].frames !== motions.baseline.frames || motions[item.id].fps !== motions.baseline.fps)) {
        throw new Error("Motion versions have inconsistent frame counts");
      }
      setSession({
        audioName: processed.audio_name,
        audioUrl: processed.audio_url,
        filename: selectedFile.name,
        voiceSpeed: processed.voice_speed,
        transcription: processed.transcription,
        visemeSegments: processed.viseme_segments,
        motions
      });
      setStage("ready");
    } catch (error) {
      setStage("error");
      setProcessingError(getErrorMessage(error));
    } finally {
      processingRef.current = false;
    }
  }

  function selectMode(next: MotionMode) {
    if (!session || isProcessing || isExporting) return;
    audioRef.current?.pause();
    if (audioRef.current) audioRef.current.currentTime = 0;
    setMode(next);
    setIsPlaying(false);
    setCurrentTime(0);
    setFrameIndex(0);
    setExportResult(null);
    setExportStage("idle");
    setExportError("");
  }

  async function play() {
    const audio = audioRef.current;
    if (!audio || !motion || isExporting) return;
    try {
      if (audio.ended || audio.currentTime >= duration) {
        audio.currentTime = 0;
        updatePosition(0, motion);
      }
      audio.playbackRate = 1;
      await audio.play();
    } catch (error) {
      setProcessingError(getErrorMessage(error));
    }
  }

  function pause() {
    if (!isExporting) audioRef.current?.pause();
  }

  function seek(seconds: number) {
    const audio = audioRef.current;
    if (!audio || !motion || isExporting) return;
    const time = Math.max(0, Math.min(duration, seconds));
    audio.currentTime = time;
    updatePosition(time, motion);
  }

  function reset() {
    const audio = audioRef.current;
    if (!audio || !motion || isExporting) return;
    audio.pause();
    audio.currentTime = 0;
    updatePosition(0, motion);
  }

  async function exportVideo() {
    const audio = audioRef.current;
    const canvas = rendererRef.current?.domElement;
    const activeMotion = motion;
    const activeSession = session;
    if (!audio || !canvas || !activeMotion || !activeSession || exportingRef.current) return;
    const mime = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find(
      (value) => typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(value)
    );
    if (!mime || typeof canvas.captureStream !== "function") {
      setExportStage("error");
      setExportError("WebM canvas recording is unsupported. Use Chrome.");
      return;
    }
    exportingRef.current = true;
    setExportStage("recording");
    setExportProgress(0);
    setExportError("");
    setExportResult(null);
    let stream: MediaStream | null = null;
    let recorder: MediaRecorder | null = null;
    let watchdog: number | undefined;
    try {
      audio.pause();
      audio.currentTime = 0;
      audio.playbackRate = 1;
      updatePosition(0, activeMotion);
      if (audio.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        await new Promise<void>((resolve, reject) => {
          const ready = () => { cleanup(); resolve(); };
          const failed = () => { cleanup(); reject(new Error("Audio could not load")); };
          const cleanup = () => {
            audio.removeEventListener("loadeddata", ready);
            audio.removeEventListener("error", failed);
          };
          audio.addEventListener("loadeddata", ready);
          audio.addEventListener("error", failed);
          audio.load();
        });
      }
      stream = canvas.captureStream(30);
      if (!stream.getVideoTracks().length) throw new Error("Unable to capture avatar viewport");
      recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 5_000_000 });
      const activeRecorder = recorder;
      const chunks: Blob[] = [];
      const finished = new Promise<Blob>((resolve, reject) => {
        activeRecorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
        activeRecorder.onerror = () => reject(new Error("Video recording failed"));
        activeRecorder.onstop = () => {
          const blob = new Blob(chunks, { type: "video/webm" });
          if (!blob.size) reject(new Error("Recorded video is empty"));
          else resolve(blob);
        };
      });
      const stop = () => { if (activeRecorder.state !== "inactive") activeRecorder.stop(); };
      audio.addEventListener("ended", stop, { once: true });
      try {
        activeRecorder.start(250);
        watchdog = window.setTimeout(stop, Math.ceil((duration + 5) * 1000));
        await audio.play();
        const recording = await finished;
        if (audio.currentTime < duration - 0.5) throw new Error("Recording ended before audio finished");
        setExportStage("processing");
        const body = new FormData();
        body.append("recording", recording, "avatar-recording.webm");
        body.append("audio_name", activeSession.audioName);
        const response = await fetch(`${API}/export-video`, { method: "POST", body });
        const result = await readResponse<ExportResponse>(response);
        setExportResult(result);
        setExportStage("completed");
        setExportProgress(100);
      } finally {
        audio.removeEventListener("ended", stop);
      }
    } catch (error) {
      if (recorder && (recorder as MediaRecorder).state !== "inactive") (recorder as MediaRecorder).stop();
      setExportStage("error");
      setExportError(getErrorMessage(error));
    } finally {
      if (watchdog !== undefined) window.clearTimeout(watchdog);
      stream?.getTracks().forEach((track) => track.stop());
      audio.pause();
      audio.currentTime = 0;
      updatePosition(0, activeMotion);
      setIsPlaying(false);
      exportingRef.current = false;
    }
  }

  const parameters = [
    { label: "Mouth Open", value: frame[0] },
    { label: "Mouth Width", value: frame[1] },
    { label: "Brow Raise", value: frame[2] },
    { label: "Gesture Intensity", value: frame[6] }
  ];

  return (
    <main className="studio">
      <header className="studio-header">
        <div className="brand">
          <div className="brand-symbol">DH</div>
          <div className="brand-text"><strong>Digital Human</strong><span>Animation Studio</span></div>
        </div>
        <div className="header-status"><span className="status-dot" /><span>Development Workspace</span></div>
      </header>
      <section className="studio-content">
        <div className="intro">
          <p className="section-label">AUDIO-DRIVEN ANIMATION</p>
          <h1>Bring motion to life.</h1>
          <p>Upload audio, select a natural speech pace, preview your digital human and export video.</p>
        </div>

        <section className="audio-upload-panel">
          <div className="audio-upload-heading">
            <div>
              <span className="panel-kicker">AUDIO SOURCE</span>
              <h2>New animation</h2>
              <p>WAV, MP3, M4A or AAC. The first 15 seconds of the source will be processed.</p>
            </div>
            <span className="upload-format-tag">MAX 25 MB</span>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.m4a,.aac"
            className="audio-file-input"
            onChange={handleFileChange}
            disabled={isProcessing || isExporting}
          />
          <div className="audio-upload-actions">
            <button type="button" className="audio-select-button" onClick={() => fileInputRef.current?.click()} disabled={isProcessing || isExporting}>Choose Audio</button>
            <div className="selected-audio-name">{selectedFile ? selectedFile.name : "No audio file selected"}</div>
            <button type="button" className="audio-process-button" onClick={handleProcess} disabled={!selectedFile || isProcessing || isExporting}>
              {isProcessing ? "Processing..." : "Generate Animation"}
            </button>
          </div>
          <div className="voice-speed-control">
            <label htmlFor="voice-speed">Voice speed</label>
            <select
              id="voice-speed"
              value={voiceSpeed}
              onChange={(event) => setVoiceSpeed(Number(event.target.value) as VoiceSpeed)}
              disabled={isProcessing || isExporting}
            >
              {SPEED_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
            </select>
            <p>Speed is applied to the audio before motion generation. Preview and MP4 both use this processed recording.</p>
          </div>
          {isProcessing && (
            <div className="audio-processing-status">
              <span className="audio-processing-indicator" />
              {stage === "uploading" ? "Uploading audio..." : stage === "processing" ? "Adjusting voice speed and generating facial motion..." : "Loading motion sequences..."}
            </div>
          )}
          {processingError && <p className="audio-processing-error">{processingError}</p>}
          {session && (
            <div className="audio-session-details">
              <div><span>ACTIVE AUDIO</span><strong>{session.filename}</strong></div>
              <div><span>VOICE SPEED</span><strong>{Math.round(session.voiceSpeed * 100)}%</strong></div>
              <div><span>DURATION</span><strong>{formatTime(duration)}</strong></div>
              <div><span>VISEME SEGMENTS</span><strong>{session.visemeSegments}</strong></div>
            </div>
          )}
        </section>

        <section className="motion-compare">
          <div className="compare-intro">
            <span className="panel-kicker">ANIMATION MODE</span>
            <h2>Motion comparison</h2>
            <p>Compare baseline, phoneme-aware and smoothed facial motion at the selected speech speed.</p>
          </div>
          <div className="mode-switch" role="group" aria-label="Motion mode">
            {MODES.map((item) => (
              <button key={item.id} type="button" className={mode === item.id ? "mode-option active" : "mode-option"} onClick={() => selectMode(item.id)} disabled={!session || isExporting || isProcessing} aria-pressed={mode === item.id}>{item.label}</button>
            ))}
          </div>
        </section>

        <div className="studio-grid">
          <section className="preview-panel">
            <div className="panel-header">
              <div><span className="panel-kicker">REAL-TIME VIEWPORT</span><h2>3D Avatar</h2></div>
              <span className="preview-tag">{MODES.find((item) => item.id === mode)?.label}</span>
            </div>
            <div className="avatar-viewport">
              <Canvas camera={{ position: [-1.8, 0.8, 3], fov: 45, near: 0.1, far: 100 }} onCreated={({ gl }) => { rendererRef.current = gl; }}>
                <ambientLight intensity={2} />
                <directionalLight position={[3, 5, 5]} intensity={3} />
                <directionalLight position={[-3, 2, -2]} intensity={1.5} />
                <Suspense fallback={null}><AvatarModel frame={frame} /></Suspense>
                <OrbitControls target={[0, 0.15, -0.2]} enableDamping enablePan={false} minDistance={1.5} maxDistance={6} />
              </Canvas>
              <div className="viewport-label"><span className="viewport-dot" />MOTION PREVIEW</div>
            </div>
            <div className="timeline-section">
              <div className="timeline-heading">
                <div><span className="panel-kicker">AUDIO TIMELINE</span><h3>Motion Sequence</h3></div>
                <span className="timeline-duration">{formatTime(currentTime)} / {formatTime(duration)}</span>
              </div>
              <div className="waveform-container">
                <div className="waveform-bars">
                  {waveform.length ? waveform.map((height, index) => (
                    <span key={index} className={((index + 0.5) / waveform.length * 100) <= progress ? "waveform-bar played" : "waveform-bar"} style={{ height: `${Math.max(8, height * 100)}%` }} />
                  )) : <span className="waveform-placeholder">{session ? "Loading waveform..." : "Upload audio to begin"}</span>}
                </div>
                <div className="timeline-playhead" style={{ left: `${progress}%` }} />
                <input className="timeline-range" type="range" min={0} max={duration || 1} step={0.01} value={currentTime} onChange={(event) => seek(Number(event.target.value))} disabled={!motion || isExporting} aria-label="Seek animation" />
              </div>
              <div className="timeline-markers">{[0, 0.25, 0.5, 0.75, 1].map((position) => <span key={position}>{formatTime(duration * position)}</span>)}</div>
              <div className="transport-controls">
                <div className="transport-buttons">
                  <button type="button" className="icon-button" onClick={() => seek(currentTime - 5)} disabled={!motion || isExporting}>−5s</button>
                  <button type="button" className="primary-button" onClick={isPlaying ? pause : play} disabled={!motion || isExporting}>{isPlaying ? "Pause" : "Play"}</button>
                  <button type="button" className="icon-button" onClick={() => seek(currentTime + 5)} disabled={!motion || isExporting}>+5s</button>
                  <button type="button" className="secondary-button" onClick={reset} disabled={!motion || isExporting}>Reset</button>
                </div>
                <div className="frame-indicator"><span>FRAME</span><strong>{motion ? String(frameIndex + 1).padStart(3, "0") : "000"}</strong><span>/ {motion?.frames ?? 0}</span></div>
              </div>
            </div>
            <section className="export-section">
              <div className="export-heading">
                <div><span className="panel-kicker">VIDEO OUTPUT</span><h3>Export Animation</h3><p>Export the selected motion with the same processed speech audio.</p></div>
                <span className="export-format">MP4</span>
              </div>
              <div className="export-actions">
                <button type="button" className="export-button" onClick={exportVideo} disabled={!motion || isExporting}>
                  {exportStage === "recording" ? "Recording..." : exportStage === "processing" ? "Encoding MP4..." : "Export MP4"}
                </button>
                {exportResult && <a className="download-button" href={`${API}${exportResult.download_url}`} download>Download Video</a>}
              </div>
              {isExporting && (
                <div className="export-status">
                  <div className="export-status-top"><span>{exportStage === "recording" ? "Capturing animation" : "Encoding with FFmpeg"}</span><strong>{exportStage === "recording" ? `${Math.round(exportProgress)}%` : "Processing"}</strong></div>
                  <div className="export-progress"><div className={exportStage === "processing" ? "export-progress-fill processing" : "export-progress-fill"} style={{ width: exportStage === "processing" ? "100%" : `${exportProgress}%` }} /></div>
                </div>
              )}
              {exportStage === "completed" && <p className="export-success">MP4 exported successfully.</p>}
              {exportError && <p className="export-error">{exportError}</p>}
            </section>
          </section>
          <aside className="details-panel">
            <div className="details-header"><span className="panel-kicker">MOTION INSPECTOR</span><h2>Animation Data</h2><p>Inspect the selected animation at the current frame.</p></div>
            {([
              ["Mode", MODES.find((item) => item.id === mode)?.label ?? "—"],
              ["Voice Speed", session ? `${Math.round(session.voiceSpeed * 100)}%` : "—"],
              ["Frames", motion?.frames ?? "—"],
              ["Frame Rate", motion ? `${motion.fps} FPS` : "—"],
              ["Dimensions", motion?.parameters.length ?? "—"],
              ["Current Frame", motion ? frameIndex + 1 : "—"]
            ] as [string, string | number][]).map(([label, value]) => <div className="data-item" key={label}><span>{label}</span><strong>{value}</strong></div>)}
            <div className="motion-heading"><span className="panel-kicker">ACTIVE PARAMETERS</span></div>
            {parameters.map((parameter) => {
              const value = Math.max(0, Math.min(1, parameter.value ?? 0));
              return <div className="parameter" key={parameter.label}><div className="parameter-top"><span>{parameter.label}</span><strong>{(value * 100).toFixed(0)}%</strong></div><div className="parameter-track"><div className="parameter-fill" style={{ width: `${value * 100}%` }} /></div></div>;
            })}
            <div className="difference-panel">
              <span className="panel-kicker">MOTION ANALYSIS</span>
              {statistics && ([
                ["Mean Mouth Opening", statistics.mean],
                ["Mean Frame Change", statistics.meanChange],
                ["Maximum Frame Change", statistics.maxChange],
                ["Maximum Mouth Opening", statistics.maxOpening]
              ] as [string, number][]).map(([label, value]) => <div className="difference-row" key={label}><span>{label}</span><strong>{value.toFixed(5)}</strong></div>)}
              <p>Metrics describe motion changes, not verified lip-sync accuracy.</p>
            </div>
            <div className="inspector-footer"><span className="status-dot" />{session ? "Motion data ready" : "Waiting for audio"}</div>
          </aside>
        </div>
        <p className="development-note">Voice speed is applied once during processing. The player runs at 100% of that refined audio, keeping motion and MP4 synchronized.</p>
      </section>
      {session && (
        <audio key={session.audioName} ref={audioRef} preload="auto" src={`${API}${session.audioUrl}`} onPlay={() => setIsPlaying(true)} onPause={() => setIsPlaying(false)} onEnded={() => { setIsPlaying(false); if (motion) updatePosition(duration, motion); }} onTimeUpdate={(event) => { if (motion) updatePosition(event.currentTarget.currentTime, motion); }} />
      )}
    </main>
  );
}

export default App;
