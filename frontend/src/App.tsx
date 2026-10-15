
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

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

const MOTION_FILE =
  "fcee2e9c479544a3b0d8186e421a8eff_15s_motion.npy";

const AUDIO_FILE =
  "fcee2e9c479544a3b0d8186e421a8eff_15s.wav";

function App() {
  const [motion, setMotion] = useState<MotionResponse | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(true);

  const audioRef = useRef<HTMLAudioElement>(null);

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
          result.frames === 0 ||
          result.fps <= 0
        ) {
          throw new Error("Invalid motion data");
        }

        setMotion(result);
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(
            err instanceof Error
              ? err.message
              : "Unable to load animation"
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
    if (!isPlaying || !motion) {
      return;
    }

    let animationId = 0;

    const updateFrame = () => {
      const audio = audioRef.current;

      if (!audio || audio.paused || audio.ended) {
        return;
      }

      const nextFrame = Math.min(
        Math.floor(audio.currentTime * motion.fps),
        motion.frames - 1
      );

      setFrameIndex(nextFrame);

      animationId = requestAnimationFrame(updateFrame);
    };

    animationId = requestAnimationFrame(updateFrame);

    return () => cancelAnimationFrame(animationId);
  }, [isPlaying, motion]);

  const handlePlay = async () => {
    if (!motion || !audioRef.current) {
      return;
    }

    try {
      await audioRef.current.play();
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
    if (!audioRef.current) {
      return;
    }

    audioRef.current.pause();
    audioRef.current.currentTime = 0;

    setFrameIndex(0);
    setIsPlaying(false);
  };

  const currentFrame =
    motion?.data[frameIndex] ?? [0, 0, 0, 0, 0, 0, 0];

  const duration = motion
    ? motion.frames / motion.fps
    : 0;

  const currentTime = motion
    ? frameIndex / motion.fps
    : 0;

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
          <span>Avatar Integration</span>
        </div>
      </header>

      <section className="studio-content">
        <div className="intro">
          <p className="section-label">
            AUDIO-DRIVEN ANIMATION
          </p>

          <h1>Bring motion to life.</h1>

          <p>
            Transform audio-driven motion into expressive
            digital human animation.
          </p>
        </div>

        <div className="studio-grid">
          <div className="preview-panel">
            <div className="panel-header">
              <div>
                <span className="panel-kicker">
                  LIVE PREVIEW
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

            <div className="playback-panel">
              <div className="time-row">
                <span>{currentTime.toFixed(2)}s</span>
                <span>{duration.toFixed(2)}s</span>
              </div>

              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{
                    width: `${
                      motion
                        ? ((frameIndex + 1) /
                            motion.frames) *
                          100
                        : 0
                    }%`
                  }}
                />
              </div>

              <div className="playback-buttons">
                <button
                  type="button"
                  className="primary-button"
                  onClick={
                    isPlaying ? handlePause : handlePlay
                  }
                  disabled={!motion}
                >
                  {isPlaying
                    ? "Pause Animation"
                    : "Play Animation"}
                </button>

                <button
                  type="button"
                  className="secondary-button"
                  onClick={handleReset}
                  disabled={!motion}
                >
                  Reset
                </button>
              </div>
            </div>
          </div>

          <aside className="details-panel">
            <div className="details-header">
              <span className="panel-kicker">
                MOTION DETAILS
              </span>
              <h2>Animation Data</h2>
              <p>
                Motion coefficients generated from the
                uploaded audio segment.
              </p>
            </div>

            <div className="data-item">
              <span>Model</span>
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

            {[
              { name: "Mouth Open", value: currentFrame[0] },
              { name: "Mouth Width", value: currentFrame[1] },
              { name: "Brow Raise", value: currentFrame[2] },
              {
                name: "Gesture Intensity",
                value: currentFrame[6]
              }
            ].map((item) => (
              <div className="parameter" key={item.name}>
                <div className="parameter-top">
                  <span>{item.name}</span>
                  <strong>
                    {((item.value ?? 0) * 100).toFixed(0)}%
                  </strong>
                </div>

                <div className="parameter-track">
                  <div
                    className="parameter-fill"
                    style={{
                      width: `${
                        Math.max(
                          0,
                          Math.min(1, item.value ?? 0)
                        ) * 100
                      }%`
                    }}
                  />
                </div>
              </div>
            ))}

            {isLoading && (
              <p className="state-message">
                Loading motion data...
              </p>
            )}

            {error && (
              <p className="error-message">{error}</p>
            )}
          </aside>
        </div>

        <p className="development-note">
          Development preview using audio-derived motion
          coefficients. Realistic speech-to-motion generation
          and final video export are upcoming stages.
        </p>
      </section>

      <audio
        ref={audioRef}
        src={`${API_BASE_URL}/test-audio/${AUDIO_FILE}`}
        onEnded={handleReset}
        onPause={() => setIsPlaying(false)}
        onPlay={() => setIsPlaying(true)}
      />
    </main>
  );
}

export default App;
