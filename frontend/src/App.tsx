
import { Suspense, useEffect, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { WebGLRenderer } from "three";
import AvatarModel from "./AvatarModel";
import "./App.css";
import "./ExportVideo.css";
import "./MotionCompare.css";

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

type MotionMode = "baseline" | "viseme" | "smoothed";
type ExportStage = "idle" | "recording" | "processing" | "completed" | "error";

const API = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const PREFIX = "fcee2e9c479544a3b0d8186e421a8eff_15s";
const AUDIO_FILE = `${PREFIX}.wav`;

const MODES: {
  id: MotionMode;
  label: string;
  file: string;
}[] = [
  {
    id: "baseline",
    label: "Baseline",
    file: `${PREFIX}_motion.npy`
  },
  {
    id: "viseme",
    label: "Viseme-Enhanced",
    file: `${PREFIX}_viseme_motion.npy`
  },
  {
    id: "smoothed",
    label: "Smoothed Viseme",
    file: `${PREFIX}_smoothed_motion.npy`
  }
];

function formatTime(seconds: number) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safe / 60);
  const remainder = (safe % 60).toFixed(2).padStart(5, "0");
  return `${String(minutes).padStart(2, "0")}:${remainder}`;
}

function makeWaveform(samples: Float32Array, count = 110) {
  const values: number[] = [];

  for (let i = 0; i < count; i++) {
    const start = Math.floor((i * samples.length) / count);
    const end = Math.floor(((i + 1) * samples.length) / count);
    let power = 0;

    for (let j = start; j < end; j++) {
      power += samples[j] ** 2;
    }

    values.push(Math.sqrt(power / Math.max(1, end - start)));
  }

  const maximum = Math.max(0.0001, ...values);
  return values.map((value) => Math.max(0.06, value / maximum));
}

function getStatistics(motion?: MotionResponse) {
  if (!motion || motion.frames < 2) {
    return null;
  }

  const values = motion.data.map((frame) => frame[0]);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const changes = values.slice(1).map(
    (value, index) => Math.abs(value - values[index])
  );

  return {
    mean,
    meanChange: changes.reduce((a, b) => a + b, 0) / changes.length,
    maxChange: Math.max(...changes),
    maxOpening: Math.max(...values)
  };
}

function App() {
  const [mode, setMode] = useState<MotionMode>("baseline");
  const [motions, setMotions] = useState<
    Partial<Record<MotionMode, MotionResponse>>
  >({});
  const [frameIndex, setFrameIndex] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [waveform, setWaveform] = useState<number[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [exportStage, setExportStage] = useState<ExportStage>("idle");
  const [exportProgress, setExportProgress] = useState(0);
  const [exportError, setExportError] = useState("");
  const [exportResult, setExportResult] = useState<ExportResponse | null>(null);

  const audioRef = useRef<HTMLAudioElement>(null);
  const rendererRef = useRef<WebGLRenderer | null>(null);
  const exportingRef = useRef(false);

  const motion = motions[mode];
  const duration = motion ? motion.frames / motion.fps : 0;
  const progress = duration ? Math.min(100, currentTime / duration * 100) : 0;
  const exporting = exportStage === "recording" || exportStage === "processing";
  const currentFrame = motion?.data[frameIndex] ?? [0, 0, 0, 0, 0, 0, 0];
  const statistics = getStatistics(motion);

  useEffect(() => {
    const controller = new AbortController();

    async function loadMotions() {
      try {
        const entries = await Promise.all(
          MODES.map(async (item) => {
            const response = await fetch(
              `${API}/motion-data/${item.file}`,
              { signal: controller.signal }
            );

            if (!response.ok) {
              throw new Error(`Cannot load ${item.label} motion`);
            }

            const data: MotionResponse = await response.json();

            if (
              data.fps <= 0 ||
              data.frames <= 0 ||
              !Array.isArray(data.data) ||
              data.data.length !== data.frames ||
              data.data.some((frame) =>
                !Array.isArray(frame) ||
                frame.length !== 7 ||
                frame.some((value) => !Number.isFinite(value))
              )
            ) {
              throw new Error(`Invalid motion data: ${item.label}`);
            }

            return [item.id, data] as const;
          })
        );

        if (!controller.signal.aborted) {
          const result = Object.fromEntries(entries) as Record<
            MotionMode,
            MotionResponse
          >;

          const reference = result.baseline;

          if (
            MODES.some((item) =>
              result[item.id].frames !== reference.frames ||
              result[item.id].fps !== reference.fps
            )
          ) {
            throw new Error("Motion sequences have different frame counts or FPS");
          }

          setMotions(result);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err instanceof Error ? err.message : "Motion loading failed");
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    void loadMotions();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    async function loadWaveform() {
      let context: AudioContext | null = null;

      try {
        const response = await fetch(
          `${API}/test-audio/${AUDIO_FILE}`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error("Audio unavailable");
        }

        const buffer = await response.arrayBuffer();

        if (controller.signal.aborted) {
          return;
        }

        context = new AudioContext();
        const decoded = await context.decodeAudioData(buffer);

        if (!controller.signal.aborted) {
          setWaveform(makeWaveform(decoded.getChannelData(0)));
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
  }, []);

  const updatePosition = (time: number, active: MotionResponse) => {
    const safe = Math.min(Math.max(0, time), active.frames / active.fps);
    const frame = Math.min(active.frames - 1, Math.floor(safe * active.fps));

    setCurrentTime(safe);
    setFrameIndex(frame);
  };

  useEffect(() => {
    if (!isPlaying || !motion) {
      return;
    }

    let animationId = 0;

    const tick = () => {
      const audio = audioRef.current;

      if (!audio || audio.paused || audio.ended) {
        return;
      }

      updatePosition(audio.currentTime, motion);

      if (exportingRef.current) {
        setExportProgress(
          Math.min(100, audio.currentTime / duration * 100)
        );
      }

      animationId = requestAnimationFrame(tick);
    };

    animationId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animationId);
  }, [isPlaying, motion, duration]);

  const selectMode = (next: MotionMode) => {
    if (exporting || !motions[next]) {
      return;
    }

    audioRef.current?.pause();

    if (audioRef.current) {
      audioRef.current.currentTime = 0;
    }

    setMode(next);
    setIsPlaying(false);
    setCurrentTime(0);
    setFrameIndex(0);
    setExportResult(null);
    setExportStage("idle");
    setExportError("");
  };

  const play = async () => {
    if (!audioRef.current || !motion || exporting) {
      return;
    }

    try {
      if (audioRef.current.ended || audioRef.current.currentTime >= duration) {
        audioRef.current.currentTime = 0;
        updatePosition(0, motion);
      }

      await audioRef.current.play();
      setError("");
    } catch {
      setError("Unable to play audio");
    }
  };

  const pause = () => {
    if (!exporting) {
      audioRef.current?.pause();
    }
  };

  const seek = (seconds: number) => {
    if (!audioRef.current || !motion || exporting) {
      return;
    }

    const time = Math.min(Math.max(0, seconds), duration);
    audioRef.current.currentTime = time;
    updatePosition(time, motion);
  };

  const reset = () => {
    if (!audioRef.current || !motion || exporting) {
      return;
    }

    audioRef.current.pause();
    audioRef.current.currentTime = 0;
    updatePosition(0, motion);
    setIsPlaying(false);
  };

  const exportVideo = async () => {
    const audio = audioRef.current;
    const canvas = rendererRef.current?.domElement;

    if (!audio || !canvas || !motion || exportingRef.current) {
      return;
    }

    const mime = [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm"
    ].find((value) =>
      typeof MediaRecorder !== "undefined" &&
      MediaRecorder.isTypeSupported(value)
    );

    if (!mime || typeof canvas.captureStream !== "function") {
      setExportStage("error");
      setExportError("WebM canvas recording is not supported in this browser");
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
      updatePosition(0, motion);

      if (audio.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        await new Promise<void>((resolve, reject) => {
          const ready = () => {
            cleanup();
            resolve();
          };

          const failed = () => {
            cleanup();
            reject(new Error("Source audio failed to load"));
          };

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

      if (!stream.getVideoTracks().length) {
        throw new Error("Unable to capture the avatar viewport");
      }

      recorder = new MediaRecorder(stream, {
        mimeType: mime,
        videoBitsPerSecond: 5_000_000
      });

      const activeRecorder = recorder;
      const chunks: Blob[] = [];

      const finished = new Promise<Blob>((resolve, reject) => {
        activeRecorder.ondataavailable = (event) => {
          if (event.data.size) {
            chunks.push(event.data);
          }
        };

        activeRecorder.onerror = () => {
          reject(new Error("Browser recording failed"));
        };

        activeRecorder.onstop = () => {
          const blob = new Blob(chunks, { type: "video/webm" });

          if (!blob.size) {
            reject(new Error("Recorded video is empty"));
          } else {
            resolve(blob);
          }
        };
      });

      const stop = () => {
        if (activeRecorder.state !== "inactive") {
          activeRecorder.stop();
        }
      };

      audio.addEventListener("ended", stop, { once: true });

      try {
        activeRecorder.start(250);

        watchdog = window.setTimeout(
          stop,
          Math.ceil((duration + 5) * 1000)
        );

        await audio.play();

        const blob = await finished;

        if (audio.currentTime < duration - 0.5) {
          throw new Error("Recording ended before the audio completed");
        }

        setExportStage("processing");

        const body = new FormData();
        body.append("recording", blob, "avatar-recording.webm");
        body.append("audio_name", AUDIO_FILE);

        const response = await fetch(`${API}/export-video`, {
          method: "POST",
          body
        });

        const result = await response.json();

        if (!response.ok) {
          throw new Error(
            typeof result.detail === "string"
              ? result.detail
              : "MP4 export failed"
          );
        }

        setExportResult(result as ExportResponse);
        setExportStage("completed");
        setExportProgress(100);
      } finally {
        audio.removeEventListener("ended", stop);
      }
    } catch (err) {
      if (recorder && (recorder as MediaRecorder).state !== "inactive") {
        (recorder as MediaRecorder).stop();
      }

      setExportStage("error");
      setExportError(
        err instanceof Error ? err.message : "Video export failed"
      );
    } finally {
      if (watchdog !== undefined) {
        clearTimeout(watchdog);
      }

      stream?.getTracks().forEach((track) => track.stop());
      audio.pause();
      audio.currentTime = 0;
      updatePosition(0, motion);
      setIsPlaying(false);
      exportingRef.current = false;
    }
  };

  const parameters = [
    { label: "Mouth Open", value: currentFrame[0] },
    { label: "Mouth Width", value: currentFrame[1] },
    { label: "Brow Raise", value: currentFrame[2] },
    { label: "Gesture Intensity", value: currentFrame[6] }
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
          <p className="section-label">AUDIO-DRIVEN ANIMATION</p>
          <h1>Bring motion to life.</h1>
          <p>Explore, compare and export audio-driven facial animation.</p>
        </div>

        <section className="motion-compare">
          <div className="compare-intro">
            <span className="panel-kicker">ANIMATION MODE</span>
            <h2>Motion comparison</h2>
            <p>
              Compare original, speech-token-based and temporally
              smoothed motion on the same avatar.
            </p>
          </div>

          <div className="mode-switch" role="group" aria-label="Motion mode">
            {MODES.map((item) => (
              <button
                key={item.id}
                type="button"
                className={
                  mode === item.id
                    ? "mode-option active"
                    : "mode-option"
                }
                onClick={() => selectMode(item.id)}
                disabled={exporting || !motions[item.id]}
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
                <span className="panel-kicker">REAL-TIME VIEWPORT</span>
                <h2>3D Avatar</h2>
              </div>
              <span className="preview-tag">
                {MODES.find((item) => item.id === mode)?.label}
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
                <directionalLight position={[3, 5, 5]} intensity={3} />
                <directionalLight position={[-3, 2, -2]} intensity={1.5} />
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
                  <span className="panel-kicker">AUDIO TIMELINE</span>
                  <h3>Motion Sequence</h3>
                </div>
                <span className="timeline-duration">
                  {formatTime(currentTime)} / {formatTime(duration)}
                </span>
              </div>

              <div className="waveform-container">
                <div className="waveform-bars">
                  {waveform.length ? (
                    waveform.map((height, index) => (
                      <span
                        key={index}
                        className={
                          ((index + 0.5) / waveform.length) * 100 <= progress
                            ? "waveform-bar played"
                            : "waveform-bar"
                        }
                        style={{ height: `${Math.max(8, height * 100)}%` }}
                      />
                    ))
                  ) : (
                    <span className="waveform-placeholder">
                      Loading waveform or audio unavailable
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
                  min={0}
                  max={duration || 1}
                  step={0.01}
                  value={currentTime}
                  onChange={(event) => seek(Number(event.target.value))}
                  disabled={!motion || exporting}
                  aria-label="Seek animation"
                />
              </div>

              <div className="timeline-markers">
                {[0, 0.25, 0.5, 0.75, 1].map((position) => (
                  <span key={position}>
                    {formatTime(duration * position)}
                  </span>
                ))}
              </div>

              <div className="transport-controls">
                <div className="transport-buttons">
                  <button
                    type="button"
                    className="icon-button"
                    onClick={() => seek(currentTime - 5)}
                    disabled={!motion || exporting}
                  >
                    −5s
                  </button>
                  <button
                    type="button"
                    className="primary-button"
                    onClick={isPlaying ? pause : play}
                    disabled={!motion || exporting}
                  >
                    {isPlaying ? "Pause" : "Play"}
                  </button>
                  <button
                    type="button"
                    className="icon-button"
                    onClick={() => seek(currentTime + 5)}
                    disabled={!motion || exporting}
                  >
                    +5s
                  </button>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={reset}
                    disabled={!motion || exporting}
                  >
                    Reset
                  </button>
                </div>

                <div className="frame-indicator">
                  <span>FRAME</span>
                  <strong>
                    {motion ? String(frameIndex + 1).padStart(3, "0") : "000"}
                  </strong>
                  <span>/ {motion?.frames ?? 0}</span>
                </div>
              </div>
            </div>

            <section className="export-section">
              <div className="export-heading">
                <div>
                  <span className="panel-kicker">VIDEO OUTPUT</span>
                  <h3>Export Animation</h3>
                  <p>
                    Record the selected motion mode and combine it
                    with the original audio.
                  </p>
                </div>
                <span className="export-format">MP4</span>
              </div>

              <div className="export-actions">
                <button
                  type="button"
                  className="export-button"
                  onClick={exportVideo}
                  disabled={!motion || exporting}
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

              {exporting && (
                <div className="export-status">
                  <div className="export-status-top">
                    <span>
                      {exportStage === "recording"
                        ? "Capturing animation"
                        : "Encoding with FFmpeg"}
                    </span>
                    <strong>
                      {exportStage === "recording"
                        ? `${Math.round(exportProgress)}%`
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
                        width: exportStage === "processing"
                          ? "100%"
                          : `${exportProgress}%`
                      }}
                    />
                  </div>
                </div>
              )}

              {exportStage === "completed" && (
                <p className="export-success">MP4 exported successfully.</p>
              )}

              {exportError && (
                <p className="export-error">{exportError}</p>
              )}
            </section>
          </section>

          <aside className="details-panel">
            <div className="details-header">
              <span className="panel-kicker">MOTION INSPECTOR</span>
              <h2>Animation Data</h2>
              <p>Inspect the selected motion sequence at each frame.</p>
            </div>

            {[
              ["Mode", MODES.find((item) => item.id === mode)?.label ?? "—"],
              ["Frames", motion?.frames ?? "—"],
              ["Frame Rate", motion ? `${motion.fps} FPS` : "—"],
              ["Dimensions", motion?.parameters.length ?? "—"],
              ["Current Frame", motion ? frameIndex + 1 : "—"]
            ].map(([label, value]) => (
              <div className="data-item" key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}

            <div className="motion-heading">
              <span className="panel-kicker">ACTIVE PARAMETERS</span>
            </div>

            {parameters.map((parameter) => {
              const value = Math.min(
                1,
                Math.max(0, parameter.value ?? 0)
              );

              return (
                <div className="parameter" key={parameter.label}>
                  <div className="parameter-top">
                    <span>{parameter.label}</span>
                    <strong>{(value * 100).toFixed(0)}%</strong>
                  </div>
                  <div className="parameter-track">
                    <div
                      className="parameter-fill"
                      style={{ width: `${value * 100}%` }}
                    />
                  </div>
                </div>
              );
            })}

            <div className="difference-panel">
              <span className="panel-kicker">MOTION ANALYSIS</span>

              {statistics && (
                <>
                  {[
                    ["Mean Mouth Opening", statistics.mean],
                    ["Mean Frame Change", statistics.meanChange],
                    ["Maximum Frame Change", statistics.maxChange],
                    ["Maximum Mouth Opening", statistics.maxOpening]
                  ].map(([label, value]) => (
                    <div className="difference-row" key={label}>
                      <span>{label}</span>
                      <strong>{Number(value).toFixed(5)}</strong>
                    </div>
                  ))}
                </>
              )}

              <p>
                These values describe motion behavior. They do not
                measure speech accuracy or lip-sync quality.
              </p>
            </div>

            <div className="inspector-footer">
              <span className="status-dot" />
              {loading
                ? "Loading motion sequences"
                : motion
                  ? "Motion data ready"
                  : "Motion data unavailable"}
            </div>

            {error && <p className="error-message">{error}</p>}
          </aside>
        </div>

        <p className="development-note">
          All three modes use experimental audio-derived motion.
          Smoothness measurements do not establish lip-sync accuracy.
        </p>
      </section>

      <audio
        ref={audioRef}
        preload="auto"
        src={`${API}/test-audio/${AUDIO_FILE}`}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => {
          setIsPlaying(false);
          if (motion) {
            updatePosition(duration, motion);
          }
        }}
        onTimeUpdate={(event) => {
          if (motion) {
            updatePosition(event.currentTarget.currentTime, motion);
          }
        }}
      />
    </main>
  );
}

export default App;
