import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useEffect, useRef, useState } from "react";
import DigitalHuman from "./DigitalHuman";
import "./App.css";

type MotionResponse = {
  motion_name: string;
  fps: number;
  frames: number;
  parameters: string[];
  data: number[][];
};

const MOTION_FILE =
  "fcee2e9c479544a3b0d8186e421a8eff_15s_motion.npy";

const AUDIO_FILE =
  "fcee2e9c479544a3b0d8186e421a8eff_15s.wav";

function App() {
  const [motion, setMotion] = useState<MotionResponse | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const audioRef = useRef<HTMLAudioElement>(null);
  const animationFrameRef = useRef<number | null>(null);

  useEffect(() => {
    const loadMotion = async () => {
      const response = await fetch(
        `${import.meta.env.VITE_API_BASE_URL}/motion-data/${MOTION_FILE}`
      );

      const data: MotionResponse = await response.json();

      setMotion(data);
    };

    loadMotion();
  }, []);

  useEffect(() => {
    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  const updateAnimationFrame = () => {
    if (!audioRef.current || !motion) {
      return;
    }

    const currentTime = audioRef.current.currentTime;

    const nextFrame = Math.min(
      Math.floor(currentTime * motion.fps),
      motion.frames - 1
    );

    setFrameIndex(nextFrame);

    if (!audioRef.current.paused && !audioRef.current.ended) {
      animationFrameRef.current = requestAnimationFrame(
        updateAnimationFrame
      );
    }
  };

  const handlePlay = async () => {
    if (!motion || !audioRef.current) {
      return;
    }

    await audioRef.current.play();

    setIsPlaying(true);

    animationFrameRef.current = requestAnimationFrame(
      updateAnimationFrame
    );
  };

  const handlePause = () => {
    if (!audioRef.current) {
      return;
    }

    audioRef.current.pause();
    setIsPlaying(false);

    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
    setFrameIndex(0);

    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  };

  const currentFrame =
    motion?.data[frameIndex] ?? [0, 0, 0, 0, 0, 0, 0];

  return (
    <main className="app">
      <section className="workspace">
        <div className="heading">
          <p>Audio-Driven Motion Generation</p>

          <h1>Digital Human Animation</h1>

          <span>
            Motion-driven avatar preview synchronized with audio.
          </span>
        </div>

        <div className="viewer">
          <Canvas camera={{ position: [0, 0, 6], fov: 42 }}>
            <ambientLight intensity={1.7} />

            <directionalLight
              position={[4, 5, 6]}
              intensity={2.5}
            />

            <directionalLight
              position={[-4, 2, 2]}
              intensity={1.2}
            />

            <DigitalHuman frame={currentFrame} />

            <OrbitControls
              enablePan={false}
              minDistance={4}
              maxDistance={8}
            />
          </Canvas>
        </div>

        <div className="controls">
          <div>
            <span>Frame</span>
            <strong>
              {frameIndex + 1} / {motion?.frames ?? 0}
            </strong>
          </div>

          <div>
            <span>FPS</span>
            <strong>{motion?.fps ?? 0}</strong>
          </div>

          <button
            type="button"
            onClick={isPlaying ? handlePause : handlePlay}
            disabled={!motion}
          >
            {isPlaying ? "Pause Animation" : "Play Animation"}
          </button>
        </div>

        <audio
          ref={audioRef}
          src={`${import.meta.env.VITE_API_BASE_URL}/test-audio/${AUDIO_FILE}`}
          onEnded={handleAudioEnded}
        />
      </section>
    </main>
  );
}

export default App;