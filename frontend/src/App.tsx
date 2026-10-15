
import { Suspense, useEffect, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { WebGLRenderer } from "three";
import AvatarModel from "./AvatarModel";
import "./App.css";
import "./ExportVideo.css";

type MotionResponse = {
  motion_name: string;
  fps: number;
  frames: number;
  parameters: string[];
  data: number[][];
};

type ExportResponse = {
  message: string;
  video_name: string;
  size_bytes: number;
  download_url: string;
};

type ExportStage =
  | "idle"
  | "recording"
  | "processing"
  | "completed"
  | "error";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000";

const MOTION_FILE =
  "fcee2e9c479544a3b0d8186e421a8eff_15s_motion.npy";

const AUDIO_FILE =
  "fcee2e9c479544a3b0d8186e421a8eff_15s.wav";

const WAVEFORM_BARS = 110;

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds)) {
    return "00:00.00";
  }

  const minutes = Math.floor(seconds / 60);
  const remaining = (seconds % 60).toFixed(2).padStart(5, "0");

  return `${String(minutes).padStart(2, "0")}:${remaining}`;
}

function createWaveform(
  samples: Float32Array,
  count: number
) {
  const peaks: number[] = [];

  for (let i = 0; i < count; i++) {
    const start = Math.floor(
      (i * samples.length) / count
    );

    const end = Math.floor(
      ((i + 1) * samples.length) / count
    );

    let sum = 0;

    for (let j = start; j < end; j++) {
      sum += samples[j] * samples[j];
    }

    const length = Math.max(1, end - start);
    peaks.push(Math.sqrt(sum / length));
  }

  const maximum = Math.max(...peaks, 0.0001);

  return peaks.map((peak) =>
    Math.max(0.06, Math.min(1, peak / maximum))
  );
}

function App() {
  const [motion, setMotion] = useState<MotionResponse | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [waveformLoading, setWaveformLoading] = useState(true);
  const [waveform, setWaveform] = useState<number[]>([]);
  const [error, setError] = useState("");

  const [exportStage, setExportStage] =
    useState<ExportStage>("idle");

  const [exportProgress, setExportProgress] = useState(0);
  const [exportError, setExportError] = useState("");
  const [exportResult, setExportResult] =
    useState<ExportResponse | null>(null);

  const audioRef = useRef<HTMLAudioElement>(null);
  const rendererRef = useRef<WebGLRenderer | null>(null);
  const exportLockedRef = useRef(false);

  const duration = motion
    ? motion.frames / motion.fps
    : 0;

  const progress =
    duration > 0
      ? Math.min(100, (currentTime / duration) * 100)
      : 0;

  const isExporting =
    exportStage === "recording" ||
    exportStage === "processing";

  useEffect(() => {
    const controller = new AbortController();

    const loadMotion = async () => {
      try {
        const response = await fetch(
          `${API_BASE_URL}/motion-data/${MOTION_FILE}`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error("Unable to load motion data");
        }

        const result: MotionResponse = await response.json();

        if (
          !Array.isArray(result.data) ||
          result.frames <= 0 ||
          result.fps <= 0 ||
          result.data.length !== result.frames
        ) {
          throw new Error("Invalid motion data");
        }

        setMotion(result);
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(
            err instanceof Error
              ? err.message
              : "Unable to load motion data"
          );
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsLoading(false);
        }
      }
    };

    loadMotion();

    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const context = new AudioContext();

    const loadWaveform = async () => {
      try {
        const response = await fetch(
          `${API_BASE_URL}/test-audio/${AUDIO_FILE}`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error("Unable to load waveform");
        }

        const buffer = await response.arrayBuffer();

        if (controller.signal.aborted) {
          return;
        }

        const decoded = await context.decodeAudioData(buffer);

        if (controller.signal.aborted) {
          return;
        }

        setWaveform(
          createWaveform(
            decoded.getChannelData(0),
            WAVEFORM_BARS
          )
        );
      } catch {
        if (!controller.signal.aborted) {
          setWaveform([]);
        }
      } finally {
        if (!controller.signal.aborted) {
          setWaveformLoading(false);
        }

        void context.close().catch(() => {});
      }
    };

    loadWaveform();

    return () => {
      controller.abort();
      void context.close().catch(() => {});
    };
  }, []);

  const updatePosition = (
    seconds: number,
    activeMotion: MotionResponse
  ) => {
    const maximum =
      activeMotion.frames / activeMotion.fps;

    const safeTime = Math.max(
      0,
      Math.min(seconds, maximum)
    );

    const nextFrame = Math.min(
      activeMotion.frames - 1,
      Math.floor(safeTime * activeMotion.fps)
    );

    setCurrentTime(safeTime);
    setFrameIndex(nextFrame);
  };

  useEffect(() => {
    if (!isPlaying || !motion) {
      return;
    }

    let animationId = 0;

    const update = () => {
      const audio = audioRef.current;

      if (!audio || audio.paused || audio.ended) {
        return;
      }

      updatePosition(audio.currentTime, motion);

      if (exportLockedRef.current) {
        setExportProgress(
          Math.min(
            100,
            (audio.currentTime / duration) * 100
          )
        );
      }

      animationId = requestAnimationFrame(update);
    };

    animationId = requestAnimationFrame(update);

    return () => cancelAnimationFrame(animationId);
  }, [isPlaying, motion, duration]);

  const handlePlay = async () => {
    if (!motion || !audioRef.current || isExporting) {
      return;
    }

    try {
      const audio = audioRef.current;

      if (audio.ended || audio.currentTime >= duration) {
        audio.currentTime = 0;
        updatePosition(0, motion);
      }

      await audio.play();
      setError("");
    } catch {
      setError("Unable to play the audio file");
    }
  };

  const handlePause = () => {
    if (isExporting) {
      return;
    }

    audioRef.current?.pause();
  };

  const handleReset = () => {
    if (!motion || !audioRef.current || isExporting) {
      return;
    }

    audioRef.current.pause();
    audioRef.current.currentTime = 0;

    updatePosition(0, motion);
    setIsPlaying(false);
  };

  const handleSeek = (seconds: number) => {
    if (!motion || !audioRef.current || isExporting) {
      return;
    }

    const safeTime = Math.max(
      0,
      Math.min(seconds, duration)
    );

    audioRef.current.currentTime = safeTime;
    updatePosition(safeTime, motion);
  };

  const skipTime = (seconds: number) => {
    if (!audioRef.current) {
      return;
    }

    handleSeek(
      audioRef.current.currentTime + seconds
    );
  };

  const handleExport = async () => {
    const audio = audioRef.current;
    const canvas = rendererRef.current?.domElement;

    if (
      !motion ||
      !audio ||
      !canvas ||
      exportLockedRef.current
    ) {
      return;
    }

    if (
      typeof MediaRecorder === "undefined" ||
      typeof canvas.captureStream !== "function"
    ) {
      setExportStage("error");
      setExportError(
        "This browser does not support canvas video recording."
      );
      return;
    }

    const mimeType = [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm"
    ].find((type) => MediaRecorder.isTypeSupported(type));

    if (!mimeType) {
      setExportStage("error");
      setExportError(
        "WebM recording is not supported. Try Chrome on your laptop."
      );
      return;
    }

    exportLockedRef.current = true;

    setExportError("");
    setExportResult(null);
    setExportProgress(0);
    setExportStage("recording");

    let stream: MediaStream | null = null;
    let recorder: MediaRecorder | null = null;

    try {
      audio.pause();
      audio.currentTime = 0;
      updatePosition(0, motion);

      if (audio.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        await new Promise<void>((resolve, reject) => {
          const onReady = () => {
            cleanup();
            resolve();
          };

          const onError = () => {
            cleanup();
            reject(new Error("Source audio could not be loaded"));
          };

          const cleanup = () => {
            audio.removeEventListener("loadeddata", onReady);
            audio.removeEventListener("error", onError);
          };

          audio.addEventListener("loadeddata", onReady);
          audio.addEventListener("error", onError);

          audio.load();
        });
      }

      stream = canvas.captureStream(30);

      if (!stream.getVideoTracks().length) {
        throw new Error("Unable to capture the 3D canvas");
      }

      recorder = new MediaRecorder(stream, {
        mimeType,
        videoBitsPerSecond: 5_000_000
      });

      const chunks: Blob[] = [];
      const activeRecorder = recorder;

      const recordingPromise = new Promise<Blob>(
        (resolve, reject) => {
          activeRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
              chunks.push(event.data);
            }
          };

          activeRecorder.onerror = () => {
            reject(new Error("Video recording failed"));
          };

          activeRecorder.onstop = () => {
            const blob = new Blob(chunks, {
              type: "video/webm"
            });

            if (blob.size === 0) {
              reject(new Error("The recorded video is empty"));
              return;
            }

            resolve(blob);
          };
        }
      );

      const endRecording = () => {
        if (activeRecorder.state !== "inactive") {
          activeRecorder.stop();
        }
      };

      audio.addEventListener("ended", endRecording, {
        once: true
      });

      try {
        activeRecorder.start(250);

        await audio.play();

        const recording = await recordingPromise;

        setExportStage("processing");

        const formData = new FormData();

        formData.append(
          "recording",
          recording,
          "avatar-recording.webm"
        );

        formData.append("audio_name", AUDIO_FILE);

        const response = await fetch(
          `${API_BASE_URL}/export-video`,
          {
            method: "POST",
            body: formData
          }
        );

        const result = await response.json();

        if (!response.ok) {
          throw new Error(
            typeof result.detail === "string"
              ? result.detail
              : "Video export failed"
          );
        }

        setExportResult(result as ExportResponse);
        setExportProgress(100);
        setExportStage("completed");
      } finally {
        audio.removeEventListener(
          "ended",
          endRecording
        );
      }
    } catch (err) {
      if (recorder && recorder.state !== "inactive") {
        recorder.stop();
      }

      setExportStage("error");
      setExportError(
        err instanceof Error
          ? err.message
          : "Unable to export video"
      );
    } finally {
      stream?.getTracks().forEach((track) => track.stop());

      audio.pause();
      audio.currentTime = 0;
      updatePosition(0, motion);

      setIsPlaying(false);
      exportLockedRef.current = false;
    }
  };

  const currentFrame =
    motion?.data[frameIndex] ??
    [0, 0, 0, 0, 0, 0, 0];

  const parameterValues = [
    { name: "Mouth Open", value: currentFrame[0] },
    { name: "Mouth Width", value: currentFrame[1] },
    { name: "Brow Raise", value: currentFrame[2] },
    {
      name: "Gesture Intensity",
      value: currentFrame[6]
    }
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
            Explore speech-driven motion through an
            interactive 3D animation workspace.
          </p>
        </div>

        <div className="studio-grid">
          <div className="preview-panel">
            <div className="panel-header">
              <div>
                <span className="panel-kicker">
                  REAL-TIME VIEWPORT
                </span>

                <h2>3D Avatar</h2>
              </div>

              <span className="preview-tag">
                FACE MODEL
              </span>
            </div>

            <div className="avatar-viewport">
              <Canvas
                camera={{
                  position: [-1.8, 0.8, 3],
                  fov: 45,
                  near: 0.1,
                  far: 100
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
                  <AvatarModel frame={currentFrame} />
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
                  {formatTime(currentTime)} /{" "}
                  {formatTime(duration)}
                </span>
              </div>

              <div className="waveform-container">
                <div className="waveform-bars">
                  {waveformLoading ? (
                    <span className="waveform-placeholder">
                      Loading audio waveform...
                    </span>
                  ) : waveform.length ? (
                    waveform.map((height, index) => {
                      const barProgress =
                        ((index + 0.5) /
                          waveform.length) *
                        100;

                      return (
                        <span
                          className={`waveform-bar ${
                            barProgress <= progress
                              ? "played"
                              : ""
                          }`}
                          key={index}
                          style={{
                            height: `${
                              Math.max(8, height * 100)
                            }%`
                          }}
                        />
                      );
                    })
                  ) : (
                    <span className="waveform-placeholder">
                      Waveform unavailable
                    </span>
                  )}
                </div>

                <div
                  className="timeline-playhead"
                  style={{ left: `${progress}%` }}
                />

                <input
                  className="timeline-range"
                  type="range"
                  min="0"
                  max={duration || 1}
                  step="0.01"
                  value={currentTime}
                  onChange={(event) =>
                    handleSeek(
                      Number(event.target.value)
                    )
                  }
                  disabled={!motion || isExporting}
                  aria-label="Seek animation timeline"
                />
              </div>

              <div className="timeline-markers">
                <span>00:00</span>
                <span>{formatTime(duration / 4)}</span>
                <span>{formatTime(duration / 2)}</span>
                <span>
                  {formatTime((duration * 3) / 4)}
                </span>
                <span>{formatTime(duration)}</span>
              </div>

              <div className="transport-controls">
                <div className="transport-buttons">
                  <button
                    className="icon-button"
                    type="button"
                    onClick={() => skipTime(-5)}
                    disabled={!motion || isExporting}
                  >
                    −5s
                  </button>

                  <button
                    className="primary-button"
                    type="button"
                    onClick={
                      isPlaying ? handlePause : handlePlay
                    }
                    disabled={!motion || isExporting}
                  >
                    {isPlaying ? "Pause" : "Play"}
                  </button>

                  <button
                    className="icon-button"
                    type="button"
                    onClick={() => skipTime(5)}
                    disabled={!motion || isExporting}
                  >
                    +5s
                  </button>

                  <button
                    className="secondary-button"
                    type="button"
                    onClick={handleReset}
                    disabled={!motion || isExporting}
                  >
                    Reset
                  </button>
                </div>

                <div className="frame-indicator">
                  <span>FRAME</span>
                  <strong>
                    {motion
                      ? String(frameIndex + 1).padStart(3, "0")
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
                    Record the animated avatar and combine
                    it with the original audio.
                  </p>
                </div>

                <span className="export-format">MP4</span>
              </div>

              <div className="export-actions">
                <button
                  type="button"
                  className="export-button"
                  onClick={handleExport}
                  disabled={!motion || isExporting}
                >
                  {exportStage === "recording"
                    ? "Recording Animation..."
                    : exportStage === "processing"
                      ? "Encoding MP4..."
                      : "Export MP4"}
                </button>

                {exportResult && (
                  <a
                    className="download-button"
                    href={`${API_BASE_URL}${exportResult.download_url}`}
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
                        ? "Capturing 3D animation"
                        : "Processing video with FFmpeg"}
                    </span>

                    <strong>
                      {exportStage === "recording"
                        ? `${Math.round(exportProgress)}%`
                        : "Encoding"}
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
                          exportStage === "recording"
                            ? `${exportProgress}%`
                            : "100%"
                      }}
                    />
                  </div>
                </div>
              )}

              {exportStage === "completed" && (
                <p className="export-success">
                  MP4 video exported successfully.
                </p>
              )}

              {exportError && (
                <p className="export-error">
                  {exportError}
                </p>
              )}
            </section>
          </div>

          <aside className="details-panel">
            <div className="details-header">
              <span className="panel-kicker">
                MOTION INSPECTOR
              </span>

              <h2>Animation Data</h2>

              <p>
                Inspect audio-derived motion parameters
                as the timeline plays.
              </p>
            </div>

            <div className="data-item">
              <span>Avatar Model</span>
              <strong>Facecap 3D</strong>
            </div>

            <div className="data-item">
              <span>Total Frames</span>
              <strong>{motion?.frames ?? "—"}</strong>
            </div>

            <div className="data-item">
              <span>Frame Rate</span>
              <strong>
                {motion ? `${motion.fps} FPS` : "—"}
              </strong>
            </div>

            <div className="data-item">
              <span>Motion Dimensions</span>
              <strong>
                {motion?.parameters.length ?? "—"}
              </strong>
            </div>

            <div className="data-item">
              <span>Current Frame</span>
              <strong>
                {motion ? frameIndex + 1 : "—"}
              </strong>
            </div>

            <div className="motion-heading">
              <span className="panel-kicker">
                ACTIVE PARAMETERS
              </span>
            </div>

            {parameterValues.map((item) => {
              const value = Math.max(
                0,
                Math.min(1, item.value ?? 0)
              );

              return (
                <div className="parameter" key={item.name}>
                  <div className="parameter-top">
                    <span>{item.name}</span>
                    <strong>
                      {(value * 100).toFixed(0)}%
                    </strong>
                  </div>

                  <div className="parameter-track">
                    <div
                      className="parameter-fill"
                      style={{
                        width: `${value * 100}%`
                      }}
                    />
                  </div>
                </div>
              );
            })}

            <div className="inspector-footer">
              <span className="status-dot" />

              {isLoading
                ? "Loading animation data"
                : motion
                  ? "Motion sequence ready"
                  : "Motion sequence unavailable"}
            </div>

            {error && (
              <p className="error-message">{error}</p>
            )}
          </aside>
        </div>

        <p className="development-note">
          Facial motion currently uses a procedural
          audio-feature baseline. Video export captures
          the live 3D preview and combines it with audio.
        </p>
      </section>

      <audio
        ref={audioRef}
        preload="auto"
        src={`${API_BASE_URL}/test-audio/${AUDIO_FILE}`}
        onEnded={() => {
          setIsPlaying(false);

          if (motion) {
            updatePosition(duration, motion);
          }
        }}
        onPause={() => setIsPlaying(false)}
        onPlay={() => setIsPlaying(true)}
        onTimeUpdate={(event) => {
          if (motion) {
            updatePosition(
              event.currentTarget.currentTime,
              motion
            );
          }
        }}
      />
    </main>
  );
}

export default App;
