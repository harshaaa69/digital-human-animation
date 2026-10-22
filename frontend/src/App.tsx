
import {
  Suspense,
  useEffect,
  useRef,
  useState,
} from "react";
import type { ChangeEvent } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { WebGLRenderer } from "three";
import AvatarModel from "./AvatarModel";
import "./App.css";
import "./AudioUpload.css";
import "./ExportVideo.css";
import "./MotionCompare.css";
import "./VoiceSpeed.css";

type MotionMode = "baseline" | "viseme" | "smoothed";
type VoiceSpeed = 1 | 0.95 | 0.9 | 0.85;
type Stage =
  | "idle"
  | "uploading"
  | "processing"
  | "loading"
  | "ready"
  | "error";

type ExportStage =
  | "idle"
  | "recording"
  | "processing"
  | "completed"
  | "error";

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
  voice_speed: number;
  transcription: string;
  viseme_segments: number;
  motion_urls: Record<MotionMode, string>;
};

type ExportResponse = {
  message: string;
  download_url: string;
  video_name: string;
  size_bytes: number;
};

type AnimationSession = {
  filename: string;
  audioName: string;
  audioUrl: string;
  voiceSpeed: number;
  transcription: string;
  visemeSegments: number;
  motions: Record<MotionMode, MotionResponse>;
};

const API =
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000";

const MODES: {
  id: MotionMode;
  label: string;
}[] = [
  { id: "baseline", label: "Baseline" },
  { id: "viseme", label: "Viseme-Enhanced" },
  { id: "smoothed", label: "Smoothed Viseme" },
];

const SPEED_OPTIONS: {
  value: VoiceSpeed;
  label: string;
}[] = [
  { value: 1, label: "100% · Original" },
  { value: 0.95, label: "95% · Gentle" },
  { value: 0.9, label: "90% · Relaxed" },
  { value: 0.85, label: "85% · Slow" },
];

const ACCEPTED_EXTENSIONS = [
  ".wav",
  ".mp3",
  ".m4a",
  ".aac",
];

const FACIAL_PARAMETERS = [
  "mouth_open",
  "mouth_width",
  "brow_raise",
  "head_yaw",
  "head_pitch",
  "head_roll",
  "gesture_intensity",
  "lip_closure",
  "lip_rounding",
  "lip_stretch",
  "lower_lip_contact",
];

function formatTime(value: number): string {
  const seconds = Number.isFinite(value)
    ? Math.max(value, 0)
    : 0;

  const minutes = Math.floor(seconds / 60);

  return `${String(minutes).padStart(2, "0")}:${(
    seconds % 60
  )
    .toFixed(2)
    .padStart(5, "0")}`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "An unexpected error occurred";
}

async function readResponse<T>(
  response: Response
): Promise<T> {
  const data: unknown = await response.json();

  if (!response.ok) {
    if (
      data &&
      typeof data === "object" &&
      "detail" in data &&
      typeof data.detail === "string"
    ) {
      throw new Error(data.detail);
    }

    throw new Error(
      `Request failed (${response.status})`
    );
  }

  return data as T;
}

function validateMotion(
  data: MotionResponse,
  mode: MotionMode
): void {
  const dimensions = mode === "baseline" ? 7 : 11;

  if (
    !Number.isFinite(data.fps) ||
    data.fps <= 0 ||
    !Number.isInteger(data.frames) ||
    data.frames <= 0 ||
    !Array.isArray(data.parameters) ||
    data.parameters.length !== dimensions ||
    data.parameters.some(
      (name, index) =>
        name !== FACIAL_PARAMETERS[index]
    ) ||
    !Array.isArray(data.data) ||
    data.data.length !== data.frames ||
    data.data.some(
      (row) =>
        !Array.isArray(row) ||
        row.length !== dimensions ||
        row.some(
          (value) => !Number.isFinite(value)
        )
    )
  ) {
    throw new Error(
      `Invalid ${dimensions}-channel data for ${mode}`
    );
  }
}

function generateWaveform(
  samples: Float32Array,
  bars = 110
): number[] {
  const levels: number[] = [];

  for (let index = 0; index < bars; index++) {
    const start = Math.floor(
      (index * samples.length) / bars
    );

    const end = Math.floor(
      ((index + 1) * samples.length) / bars
    );

    let power = 0;

    for (let offset = start; offset < end; offset++) {
      power += samples[offset] ** 2;
    }

    levels.push(
      Math.sqrt(
        power / Math.max(1, end - start)
      )
    );
  }

  const maximum = Math.max(0.0001, ...levels);

  return levels.map((level) =>
    Math.max(
      0.06,
      Math.min(1, level / maximum)
    )
  );
}

function getStatistics(motion?: MotionResponse) {
  if (!motion || motion.frames < 2) {
    return null;
  }

  const values = motion.data.map(
    (frame) => frame[0]
  );

  const changes = values.slice(1).map(
    (value, index) =>
      Math.abs(value - values[index])
  );

  return {
    mean:
      values.reduce(
        (total, value) => total + value,
        0
      ) / values.length,
    meanChange:
      changes.reduce(
        (total, value) => total + value,
        0
      ) / changes.length,
    maxChange: Math.max(...changes),
    maxOpening: Math.max(...values),
  };
}

function App() {
  const [selectedFile, setSelectedFile] =
    useState<File | null>(null);

  const [voiceSpeed, setVoiceSpeed] =
    useState<VoiceSpeed>(0.9);

  const [session, setSession] =
    useState<AnimationSession | null>(null);

  const [stage, setStage] =
    useState<Stage>("idle");

  const [processingError, setProcessingError] =
    useState("");

  const [mode, setMode] =
    useState<MotionMode>("baseline");

  const [frameIndex, setFrameIndex] =
    useState(0);

  const [currentTime, setCurrentTime] =
    useState(0);

  const [isPlaying, setIsPlaying] =
    useState(false);

  const [waveform, setWaveform] =
    useState<number[]>([]);

  const [exportStage, setExportStage] =
    useState<ExportStage>("idle");

  const [exportProgress, setExportProgress] =
    useState(0);

  const [exportError, setExportError] =
    useState("");

  const [exportResult, setExportResult] =
    useState<ExportResponse | null>(null);

  const audioRef = useRef<HTMLAudioElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const rendererRef = useRef<WebGLRenderer | null>(
    null
  );

  const exportingRef = useRef(false);
  const processingRef = useRef(false);

  const motion = session?.motions[mode];

  const duration = motion
    ? motion.frames / motion.fps
    : 0;

  const isProcessing = [
    "uploading",
    "processing",
    "loading",
  ].includes(stage);

  const isExporting =
    exportStage === "recording" ||
    exportStage === "processing";

  const frame =
    motion?.data[
      Math.min(frameIndex, motion.frames - 1)
    ] ?? Array(7).fill(0);

  const progress =
    duration > 0
      ? Math.min(
          100,
          (currentTime / duration) * 100
        )
      : 0;

  const statistics = getStatistics(motion);

  useEffect(() => {
    if (!session) {
      setWaveform([]);
      return;
    }

    const controller = new AbortController();
    let context: AudioContext | null = null;

    async function loadWaveform() {
      try {
        const response = await fetch(
          `${API}${session!.audioUrl}`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error(
            "Unable to load audio waveform"
          );
        }

        const buffer = await response.arrayBuffer();

        if (controller.signal.aborted) {
          return;
        }

        context = new AudioContext();

        const decoded =
          await context.decodeAudioData(buffer);

        if (!controller.signal.aborted) {
          setWaveform(
            generateWaveform(
              decoded.getChannelData(0)
            )
          );
        }
      } catch {
        if (!controller.signal.aborted) {
          setWaveform([]);
        }
      } finally {
        if (context) {
          void context.close().catch(() => {});
        }
      }
    }

    void loadWaveform();

    return () => controller.abort();
  }, [session]);

  function updatePosition(
    seconds: number,
    activeMotion: MotionResponse
  ) {
    const maximum =
      activeMotion.frames / activeMotion.fps;

    const time = Math.max(
      0,
      Math.min(seconds, maximum)
    );

    setCurrentTime(time);

    setFrameIndex(
      Math.min(
        activeMotion.frames - 1,
        Math.floor(time * activeMotion.fps)
      )
    );
  }

  useEffect(() => {
    if (!isPlaying || !motion) {
      return;
    }

    let animationId = 0;

    const tick = () => {
      const audio = audioRef.current;

      if (
        !audio ||
        audio.paused ||
        audio.ended
      ) {
        return;
      }

      updatePosition(
        audio.currentTime,
        motion
      );

      if (exportingRef.current && duration > 0) {
        setExportProgress(
          Math.min(
            100,
            (audio.currentTime / duration) * 100
          )
        );
      }

      animationId =
        requestAnimationFrame(tick);
    };

    animationId = requestAnimationFrame(tick);

    return () =>
      cancelAnimationFrame(animationId);
  }, [isPlaying, motion, duration]);

  function handleFileChange(
    event: ChangeEvent<HTMLInputElement>
  ) {
    const file = event.target.files?.[0] ?? null;

    if (!file) {
      return;
    }

    const extension = file.name
      .slice(file.name.lastIndexOf("."))
      .toLowerCase();

    if (
      !ACCEPTED_EXTENSIONS.includes(extension)
    ) {
      setSelectedFile(null);
      setProcessingError(
        "Unsupported audio format"
      );
      return;
    }

    if (
      file.size === 0 ||
      file.size > 25 * 1024 * 1024
    ) {
      setSelectedFile(null);
      setProcessingError(
        "Choose a non-empty audio file up to 25 MB"
      );
      return;
    }

    setSelectedFile(file);
    setProcessingError("");
  }

  async function handleProcess() {
    if (
      !selectedFile ||
      processingRef.current ||
      exportingRef.current
    ) {
      return;
    }

    processingRef.current = true;

    audioRef.current?.pause();

    setIsPlaying(false);
    setSession(null);
    setWaveform([]);
    setCurrentTime(0);
    setFrameIndex(0);
    setMode("baseline");
    setProcessingError("");
    setExportResult(null);
    setExportStage("idle");
    setExportError("");

    try {
      setStage("uploading");

      const form = new FormData();
      form.append("file", selectedFile);

      const uploaded = await readResponse<UploadResponse>(
        await fetch(`${API}/upload-audio`, {
          method: "POST",
          body: form,
        })
      );

      setStage("processing");

      const processed =
        await readResponse<ProcessingResponse>(
          await fetch(
            `${API}/process-audio/${encodeURIComponent(
              uploaded.processed_name
            )}?speed=${voiceSpeed}`,
            { method: "POST" }
          )
        );

      setStage("loading");

      const entries = await Promise.all(
        MODES.map(async (item) => {
          const response = await fetch(
            `${API}${processed.motion_urls[item.id]}`
          );

          const data =
            await readResponse<MotionResponse>(
              response
            );

          validateMotion(
            data,
            item.id
          );

          return [item.id, data] as const;
        })
      );

      const motions = Object.fromEntries(
        entries
      ) as Record<MotionMode, MotionResponse>;

      const baseline = motions.baseline;

      if (
        MODES.some(
          (item) =>
            motions[item.id].frames !==
              baseline.frames ||
            motions[item.id].fps !==
              baseline.fps
        )
      ) {
        throw new Error(
          "Motion modes have inconsistent timing"
        );
      }

      setSession({
        filename: selectedFile.name,
        audioName: processed.audio_name,
        audioUrl: processed.audio_url,
        voiceSpeed: processed.voice_speed,
        transcription: processed.transcription,
        visemeSegments: processed.viseme_segments,
        motions,
      });

      setStage("ready");
    } catch (error) {
      setStage("error");
      setProcessingError(
        errorMessage(error)
      );
    } finally {
      processingRef.current = false;
    }
  }

  function selectMode(next: MotionMode) {
    if (
      !session ||
      isProcessing ||
      isExporting
    ) {
      return;
    }

    const audio = audioRef.current;

    if (audio) {
      audio.pause();
      audio.currentTime = 0;
    }

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

    if (
      !audio ||
      !motion ||
      isExporting
    ) {
      return;
    }

    try {
      if (
        audio.ended ||
        audio.currentTime >= duration
      ) {
        audio.currentTime = 0;
        updatePosition(0, motion);
      }

      audio.playbackRate = 1;

      await audio.play();
    } catch (error) {
      setProcessingError(
        errorMessage(error)
      );
    }
  }

  function pause() {
    if (!isExporting) {
      audioRef.current?.pause();
    }
  }

  function seek(seconds: number) {
    const audio = audioRef.current;

    if (
      !audio ||
      !motion ||
      isExporting
    ) {
      return;
    }

    const time = Math.max(
      0,
      Math.min(seconds, duration)
    );

    audio.currentTime = time;
    updatePosition(time, motion);
  }

  function reset() {
    const audio = audioRef.current;

    if (
      !audio ||
      !motion ||
      isExporting
    ) {
      return;
    }

    audio.pause();
    audio.currentTime = 0;
    updatePosition(0, motion);
  }

  async function exportVideo() {
    const audio = audioRef.current;
    const canvas =
      rendererRef.current?.domElement;

    const activeMotion = motion;
    const activeSession = session;

    if (
      !audio ||
      !canvas ||
      !activeMotion ||
      !activeSession ||
      exportingRef.current
    ) {
      return;
    }

    const mime = [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm",
    ].find(
      (value) =>
        typeof MediaRecorder !== "undefined" &&
        MediaRecorder.isTypeSupported(value)
    );

    if (
      !mime ||
      typeof canvas.captureStream !== "function"
    ) {
      setExportStage("error");
      setExportError(
        "Canvas recording is unsupported. Use Chrome."
      );
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

      if (
        audio.readyState <
        HTMLMediaElement.HAVE_CURRENT_DATA
      ) {
        await new Promise<void>(
          (resolve, reject) => {
            const ready = () => {
              cleanup();
              resolve();
            };

            const failed = () => {
              cleanup();
              reject(
                new Error("Audio could not load")
              );
            };

            const cleanup = () => {
              audio.removeEventListener(
                "loadeddata",
                ready
              );

              audio.removeEventListener(
                "error",
                failed
              );
            };

            audio.addEventListener(
              "loadeddata",
              ready
            );

            audio.addEventListener(
              "error",
              failed
            );

            audio.load();
          }
        );
      }

      stream = canvas.captureStream(30);

      if (!stream.getVideoTracks().length) {
        throw new Error(
          "Unable to capture avatar viewport"
        );
      }

      recorder = new MediaRecorder(stream, {
        mimeType: mime,
        videoBitsPerSecond: 5_000_000,
      });

      const activeRecorder = recorder;
      const chunks: Blob[] = [];

      const finished = new Promise<Blob>(
        (resolve, reject) => {
          activeRecorder.ondataavailable = (
            event
          ) => {
            if (event.data.size) {
              chunks.push(event.data);
            }
          };

          activeRecorder.onerror = () =>
            reject(
              new Error("Video recording failed")
            );

          activeRecorder.onstop = () => {
            const blob = new Blob(chunks, {
              type: "video/webm",
            });

            if (!blob.size) {
              reject(
                new Error("Recorded video is empty")
              );
            } else {
              resolve(blob);
            }
          };
        }
      );

      const stop = () => {
        if (activeRecorder.state !== "inactive") {
          activeRecorder.stop();
        }
      };

      audio.addEventListener(
        "ended",
        stop,
        { once: true }
      );

      try {
        activeRecorder.start(250);

        watchdog = window.setTimeout(
          stop,
          Math.ceil(
            (duration + 5) * 1000
          )
        );

        await audio.play();

        const recording = await finished;

        if (
          audio.currentTime <
          duration - 0.5
        ) {
          throw new Error(
            "Recording ended before audio finished"
          );
        }

        setExportStage("processing");

        const form = new FormData();

        form.append(
          "recording",
          recording,
          "avatar-recording.webm"
        );

        form.append(
          "audio_name",
          activeSession.audioName
        );

        const response = await fetch(
          `${API}/export-video`,
          {
            method: "POST",
            body: form,
          }
        );

        const result =
          await readResponse<ExportResponse>(
            response
          );

        setExportResult(result);
        setExportProgress(100);
        setExportStage("completed");
      } finally {
        audio.removeEventListener(
          "ended",
          stop
        );
      }
    } catch (error) {
      if (
        recorder &&
        recorder.state !== "inactive"
      ) {
        recorder.stop();
      }

      setExportStage("error");
      setExportError(
        errorMessage(error)
      );
    } finally {
      if (watchdog !== undefined) {
        window.clearTimeout(watchdog);
      }

      stream
        ?.getTracks()
        .forEach((track) => track.stop());

      audio.pause();
      audio.currentTime = 0;

      updatePosition(0, activeMotion);

      setIsPlaying(false);
      exportingRef.current = false;
    }
  }

  const parameters = [
    {
      label: "Mouth Open",
      value: frame[0] ?? 0,
    },
    {
      label: "Mouth Width",
      value: frame[1] ?? 0,
    },
    {
      label: "Brow Raise",
      value: frame[2] ?? 0,
    },
    {
      label: "Lip Closure",
      value: frame[7] ?? 0,
    },
    {
      label: "Lip Rounding",
      value: frame[8] ?? 0,
    },
    {
      label: "Lip Stretch",
      value: frame[9] ?? 0,
    },
    {
      label: "Lower Lip Contact",
      value: frame[10] ?? 0,
    },
  ];

  return (
    <main className="studio">
      <header className="studio-header">
        <div className="brand">
          <div className="brand-symbol">DH</div>
          <div className="brand-text">
            <strong>Digital Human</strong>
            <span>Animation Studio</span>
          </div>
        </div>
        <div className="header-status">
          <span className="status-dot" />
          <span>Development Workspace</span>
        </div>
      </header>

      <section className="studio-content">
        <div className="intro">
          <p className="section-label">
            AUDIO-DRIVEN ANIMATION
          </p>
          <h1>Bring motion to life.</h1>
          <p>
            Upload audio, select a natural speech pace,
            preview your digital human and export video.
          </p>
        </div>

        <section className="audio-upload-panel">
          <div className="audio-upload-heading">
            <div>
              <span className="panel-kicker">
                AUDIO SOURCE
              </span>
              <h2>New animation</h2>
              <p>
                WAV, MP3, M4A or AAC. The first
                15 seconds of the source will be processed.
              </p>
            </div>
            <span className="upload-format-tag">
              MAX 25 MB
            </span>
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
            <button
              type="button"
              className="audio-select-button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isProcessing || isExporting}
            >
              Choose Audio
            </button>

            <div className="selected-audio-name">
              {selectedFile
                ? selectedFile.name
                : "No audio file selected"}
            </div>

            <button
              type="button"
              className="audio-process-button"
              onClick={handleProcess}
              disabled={
                !selectedFile ||
                isProcessing ||
                isExporting
              }
            >
              {isProcessing
                ? "Processing..."
                : "Generate Animation"}
            </button>
          </div>

          <div className="voice-speed-control">
            <label htmlFor="voice-speed">
              Voice speed
            </label>

            <select
              id="voice-speed"
              value={voiceSpeed}
              onChange={(event) =>
                setVoiceSpeed(
                  Number(event.target.value) as VoiceSpeed
                )
              }
              disabled={isProcessing || isExporting}
            >
              {SPEED_OPTIONS.map((option) => (
                <option
                  key={option.value}
                  value={option.value}
                >
                  {option.label}
                </option>
              ))}
            </select>

            <p>
              Speed is applied before animation generation.
              Preview and MP4 use the same processed audio.
            </p>
          </div>

          {isProcessing && (
            <div className="audio-processing-status">
              <span className="audio-processing-indicator" />
              {stage === "uploading"
                ? "Uploading audio..."
                : stage === "processing"
                ? "Generating 11-channel facial animation..."
                : "Loading motion sequences..."}
            </div>
          )}

          {processingError && (
            <p className="audio-processing-error">
              {processingError}
            </p>
          )}

          {session && (
            <div className="audio-session-details">
              <div>
                <span>ACTIVE AUDIO</span>
                <strong>{session.filename}</strong>
              </div>
              <div>
                <span>VOICE SPEED</span>
                <strong>
                  {Math.round(
                    session.voiceSpeed * 100
                  )}
                  %
                </strong>
              </div>
              <div>
                <span>DURATION</span>
                <strong>
                  {formatTime(duration)}
                </strong>
              </div>
              <div>
                <span>VISEME SEGMENTS</span>
                <strong>
                  {session.visemeSegments}
                </strong>
              </div>
            </div>
          )}
        </section>

        <section className="motion-compare">
          <div className="compare-intro">
            <span className="panel-kicker">
              ANIMATION MODE
            </span>
            <h2>Motion comparison</h2>
            <p>
              Compare seven-channel baseline motion
              with eleven-channel facial animation.
            </p>
          </div>

          <div
            className="mode-switch"
            role="group"
            aria-label="Motion mode"
          >
            {MODES.map((item) => (
              <button
                key={item.id}
                type="button"
                className={
                  mode === item.id
                    ? "mode-option active"
                    : "mode-option"
                }
                onClick={() =>
                  selectMode(item.id)
                }
                disabled={
                  !session ||
                  isExporting ||
                  isProcessing
                }
                aria-pressed={mode === item.id}
              >
                {item.label}
              </button>
            ))}
          </div>
        </section>

        <div className="studio-grid">
          <section className="preview-panel">
            <div className="panel-header">
              <div>
                <span className="panel-kicker">
                  REAL-TIME VIEWPORT
                </span>
                <h2>3D Avatar</h2>
              </div>
              <span className="preview-tag">
                {
                  MODES.find(
                    (item) => item.id === mode
                  )?.label
                }
              </span>
            </div>

            <div className="avatar-viewport">
              <Canvas
                camera={{
                  position: [-1.8, 0.8, 3],
                  fov: 45,
                  near: 0.1,
                  far: 100,
                }}
                onCreated={({ gl }) => {
                  rendererRef.current = gl;
                }}
              >
                <ambientLight intensity={2} />

                <directionalLight
                  position={[3, 5, 5]}
                  intensity={3}
                />

                <directionalLight
                  position={[-3, 2, -2]}
                  intensity={1.5}
                />

                <Suspense fallback={null}>
                  <AvatarModel frame={frame} />
                </Suspense>

                <OrbitControls
                  target={[0, 0.15, -0.2]}
                  enableDamping
                  enablePan={false}
                  minDistance={1.5}
                  maxDistance={6}
                />
              </Canvas>

              <div className="viewport-label">
                <span className="viewport-dot" />
                MOTION PREVIEW
              </div>
            </div>

            <div className="timeline-section">
              <div className="timeline-heading">
                <div>
                  <span className="panel-kicker">
                    AUDIO TIMELINE
                  </span>
                  <h3>Motion Sequence</h3>
                </div>

                <span className="timeline-duration">
                  {formatTime(currentTime)}
                  {" / "}
                  {formatTime(duration)}
                </span>
              </div>

              <div className="waveform-container">
                <div className="waveform-bars">
                  {waveform.length ? (
                    waveform.map((height, index) => (
                      <span
                        key={index}
                        className={
                          ((index + 0.5) /
                            waveform.length) *
                            100 <=
                          progress
                            ? "waveform-bar played"
                            : "waveform-bar"
                        }
                        style={{
                          height: `${Math.max(
                            8,
                            height * 100
                          )}%`,
                        }}
                      />
                    ))
                  ) : (
                    <span className="waveform-placeholder">
                      {session
                        ? "Loading waveform..."
                        : "Upload audio to begin"}
                    </span>
                  )}
                </div>

                <div
                  className="timeline-playhead"
                  style={{
                    left: `${progress}%`,
                  }}
                />

                <input
                  className="timeline-range"
                  type="range"
                  min={0}
                  max={duration || 1}
                  step={0.01}
                  value={currentTime}
                  onChange={(event) =>
                    seek(
                      Number(event.target.value)
                    )
                  }
                  disabled={!motion || isExporting}
                  aria-label="Seek animation"
                />
              </div>

              <div className="timeline-markers">
                {[0, 0.25, 0.5, 0.75, 1].map(
                  (position) => (
                    <span key={position}>
                      {formatTime(
                        duration * position
                      )}
                    </span>
                  )
                )}
              </div>

              <div className="transport-controls">
                <div className="transport-buttons">
                  <button
                    type="button"
                    className="icon-button"
                    onClick={() =>
                      seek(currentTime - 5)
                    }
                    disabled={!motion || isExporting}
                  >
                    −5s
                  </button>

                  <button
                    type="button"
                    className="primary-button"
                    onClick={
                      isPlaying
                        ? pause
                        : play
                    }
                    disabled={!motion || isExporting}
                  >
                    {isPlaying
                      ? "Pause"
                      : "Play"}
                  </button>

                  <button
                    type="button"
                    className="icon-button"
                    onClick={() =>
                      seek(currentTime + 5)
                    }
                    disabled={!motion || isExporting}
                  >
                    +5s
                  </button>

                  <button
                    type="button"
                    className="secondary-button"
                    onClick={reset}
                    disabled={!motion || isExporting}
                  >
                    Reset
                  </button>
                </div>

                <div className="frame-indicator">
                  <span>FRAME</span>
                  <strong>
                    {motion
                      ? String(
                          frameIndex + 1
                        ).padStart(3, "0")
                      : "000"}
                  </strong>
                  <span>
                    / {motion?.frames ?? 0}
                  </span>
                </div>
              </div>
            </div>

            <section className="export-section">
              <div className="export-heading">
                <div>
                  <span className="panel-kicker">
                    VIDEO OUTPUT
                  </span>
                  <h3>Export Animation</h3>
                  <p>
                    Export the selected facial motion
                    with its processed audio.
                  </p>
                </div>
                <span className="export-format">
                  MP4
                </span>
              </div>

              <div className="export-actions">
                <button
                  type="button"
                  className="export-button"
                  onClick={exportVideo}
                  disabled={!motion || isExporting}
                >
                  {exportStage === "recording"
                    ? "Recording..."
                    : exportStage === "processing"
                    ? "Encoding MP4..."
                    : "Export MP4"}
                </button>

                {exportResult && (
                  <a
                    className="download-button"
                    href={`${API}${exportResult.download_url}`}
                    download
                  >
                    Download Video
                  </a>
                )}
              </div>

              {isExporting && (
                <div className="export-status">
                  <div className="export-status-top">
                    <span>
                      {exportStage === "recording"
                        ? "Capturing animation"
                        : "Encoding with FFmpeg"}
                    </span>
                    <strong>
                      {exportStage === "recording"
                        ? `${Math.round(
                            exportProgress
                          )}%`
                        : "Processing"}
                    </strong>
                  </div>

                  <div className="export-progress">
                    <div
                      className={
                        exportStage === "processing"
                          ? "export-progress-fill processing"
                          : "export-progress-fill"
                      }
                      style={{
                        width:
                          exportStage === "processing"
                            ? "100%"
                            : `${exportProgress}%`,
                      }}
                    />
                  </div>
                </div>
              )}

              {exportStage === "completed" && (
                <p className="export-success">
                  MP4 exported successfully.
                </p>
              )}

              {exportError && (
                <p className="export-error">
                  {exportError}
                </p>
              )}
            </section>
          </section>

          <aside className="details-panel">
            <div className="details-header">
              <span className="panel-kicker">
                MOTION INSPECTOR
              </span>
              <h2>Animation Data</h2>
              <p>
                Inspect the current frame and
                additional facial controls.
              </p>
            </div>

            {(
              [
                [
                  "Mode",
                  MODES.find(
                    (item) => item.id === mode
                  )?.label ?? "—",
                ],
                [
                  "Voice Speed",
                  session
                    ? `${Math.round(
                        session.voiceSpeed * 100
                      )}%`
                    : "—",
                ],
                [
                  "Frames",
                  motion?.frames ?? "—",
                ],
                [
                  "Frame Rate",
                  motion
                    ? `${motion.fps} FPS`
                    : "—",
                ],
                [
                  "Dimensions",
                  motion?.parameters.length ?? "—",
                ],
                [
                  "Current Frame",
                  motion
                    ? frameIndex + 1
                    : "—",
                ],
              ] as [string, string | number][]
            ).map(([label, value]) => (
              <div
                className="data-item"
                key={label}
              >
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}

            <div className="motion-heading">
              <span className="panel-kicker">
                ACTIVE PARAMETERS
              </span>
            </div>

            {parameters.map((parameter) => {
              const value = Math.max(
                0,
                Math.min(1, parameter.value)
              );

              return (
                <div
                  className="parameter"
                  key={parameter.label}
                >
                  <div className="parameter-top">
                    <span>
                      {parameter.label}
                    </span>
                    <strong>
                      {(value * 100).toFixed(0)}%
                    </strong>
                  </div>

                  <div className="parameter-track">
                    <div
                      className="parameter-fill"
                      style={{
                        width: `${value * 100}%`,
                      }}
                    />
                  </div>
                </div>
              );
            })}

            <div className="difference-panel">
              <span className="panel-kicker">
                MOTION ANALYSIS
              </span>

              {statistics &&
                (
                  [
                    [
                      "Mean Mouth Opening",
                      statistics.mean,
                    ],
                    [
                      "Mean Frame Change",
                      statistics.meanChange,
                    ],
                    [
                      "Maximum Frame Change",
                      statistics.maxChange,
                    ],
                    [
                      "Maximum Mouth Opening",
                      statistics.maxOpening,
                    ],
                  ] as [string, number][]
                ).map(([label, value]) => (
                  <div
                    className="difference-row"
                    key={label}
                  >
                    <span>{label}</span>
                    <strong>
                      {value.toFixed(5)}
                    </strong>
                  </div>
                ))}

              <p>
                Motion metrics do not establish
                phoneme alignment accuracy.
              </p>
            </div>

            <div className="inspector-footer">
              <span className="status-dot" />
              {session
                ? "Motion data ready"
                : "Waiting for audio"}
            </div>
          </aside>
        </div>

        <p className="development-note">
          Eleven-channel lip controls are enabled
          for Viseme-Enhanced and Smoothed Viseme.
          The baseline remains seven-channel.
          Preview and export use the same audio.
        </p>
      </section>

      {session && (
        <audio
          key={session.audioName}
          ref={audioRef}
          preload="auto"
          src={`${API}${session.audioUrl}`}
          onPlay={() =>
            setIsPlaying(true)
          }
          onPause={() =>
            setIsPlaying(false)
          }
          onEnded={() => {
            setIsPlaying(false);

            if (motion) {
              updatePosition(
                duration,
                motion
              );
            }
          }}
          onTimeUpdate={(event) => {
            if (motion) {
              updatePosition(
                event.currentTarget.currentTime,
                motion
              );
            }
          }}
        />
      )}
    </main>
  );
}

export default App;
