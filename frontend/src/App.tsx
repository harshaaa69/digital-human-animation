
import { Suspense, useEffect, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import AvatarModel from "./AvatarModel";
import "./App.css";

type MotionResponse = {
  motion_name: string;
  fps: number;
  frames: number;
  parameters: string[];
  data: number[][];
};

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
  barCount: number
) {
  const samplesPerBar = Math.max(
    1,
    Math.floor(samples.length / barCount)
  );

  const peaks: number[] = [];

  for (let i = 0; i < barCount; i++) {
    const start = i * samplesPerBar;
    const end =
      i === barCount - 1
        ? samples.length
        : Math.min(samples.length, start + samplesPerBar);

    let sum = 0;
    let count = 0;

    for (let j = start; j < end; j++) {
      sum += samples[j] * samples[j];
      count++;
    }

    peaks.push(count ? Math.sqrt(sum / count) : 0);
  }

  const maximum = Math.max(...peaks, 0.0001);

  return peaks.map((peak) => {
    return Math.max(0.06, Math.min(1, peak / maximum));
  });
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

  const audioRef = useRef<HTMLAudioElement>(null);

  const duration = motion
    ? motion.frames / motion.fps
    : 0;

  const progress =
    duration > 0
      ? Math.min(100, (currentTime / duration) * 100)
      : 0;

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
    let context: AudioContext | null = null;

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

        context = new AudioContext();
        const decoded = await context.decodeAudioData(buffer);

        if (controller.signal.aborted) {
          return;
        }

        const samples = decoded.getChannelData(0);

        setWaveform(
          createWaveform(samples, WAVEFORM_BARS)
        );
      } catch {
        if (!controller.signal.aborted) {
          setWaveform([]);
        }
      } finally {
        if (!controller.signal.aborted) {
          setWaveformLoading(false);
        }

        if (context) {
          void context.close().catch(() => {});
        }
      }
    };

    loadWaveform();

    return () => {
      controller.abort();
    };
  }, []);

  const updatePosition = (
    seconds: number,
    activeMotion: MotionResponse
  ) => {
    const maxTime =
      activeMotion.frames / activeMotion.fps;

    const safeTime = Math.max(
      0,
      Math.min(seconds, maxTime)
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
      animationId = requestAnimationFrame(update);
    };

    animationId = requestAnimationFrame(update);

    return () => {
      cancelAnimationFrame(animationId);
    };
  }, [isPlaying, motion]);

  const handlePlay = async () => {
    const audio = audioRef.current;

    if (!audio || !motion) {
      return;
    }

    try {
      if (audio.ended || audio.currentTime >= duration) {
        audio.currentTime = 0;
        updatePosition(0, motion);
      }

      await audio.play();
      setIsPlaying(true);
      setError("");
    } catch {
      setError("Unable to play the audio file");
    }
  };

  const handlePause = () => {
    audioRef.current?.pause();
    setIsPlaying(false);
  };

  const handleReset = () => {
    const audio = audioRef.current;

    if (!audio || !motion) {
      return;
    }

    audio.pause();
    audio.currentTime = 0;

    updatePosition(0, motion);
    setIsPlaying(false);
  };

  const handleSeek = (value: number) => {
    const audio = audioRef.current;

    if (!audio || !motion) {
      return;
    }

    const safeTime = Math.max(
      0,
      Math.min(value, duration)
    );

    audio.currentTime = safeTime;
    updatePosition(safeTime, motion);
  };

  const skipTime = (seconds: number) => {
    if (!motion || !audioRef.current) {
      return;
    }

    handleSeek(audioRef.current.currentTime + seconds);
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
                    handleSeek(Number(event.target.value))
                  }
                  disabled={!motion}
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
                    disabled={!motion}
                    title="Back 5 seconds"
                  >
                    <span>−5s</span>
                  </button>

                  <button
                    className="primary-button"
                    type="button"
                    onClick={
                      isPlaying ? handlePause : handlePlay
                    }
                    disabled={!motion}
                  >
                    {isPlaying ? "Pause" : "Play"}
                  </button>

                  <button
                    className="icon-button"
                    type="button"
                    onClick={() => skipTime(5)}
                    disabled={!motion}
                    title="Forward 5 seconds"
                  >
                    <span>+5s</span>
                  </button>

                  <button
                    className="secondary-button"
                    type="button"
                    onClick={handleReset}
                    disabled={!motion}
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
              <strong>
                {motion?.frames ?? "—"}
              </strong>
            </div>

            <div className="data-item">
              <span>Frame Rate</span>
              <strong>
                {motion
                  ? `${motion.fps} FPS`
                  : "—"}
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
                <div
                  className="parameter"
                  key={item.name}
                >
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
          Facial motion is currently driven by a
          procedural audio-feature baseline. The
          timeline supports synchronized playback
          and frame-level inspection.
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
