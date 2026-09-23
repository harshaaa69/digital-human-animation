import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import type { Group } from "three";

type MotionFrame = number[];

type DigitalHumanProps = {
  frame: MotionFrame;
};

function DigitalHuman({ frame }: DigitalHumanProps) {
  const headRef = useRef<Group>(null);

  const mouthOpen = frame[0] ?? 0;
  const mouthWidth = frame[1] ?? 0;
  const browRaise = frame[2] ?? 0;
  const headYaw = frame[3] ?? 0;
  const headPitch = frame[4] ?? 0;
  const headRoll = frame[5] ?? 0;

  useFrame(() => {
    if (!headRef.current) {
      return;
    }

    headRef.current.rotation.y = headYaw;
    headRef.current.rotation.x = headPitch;
    headRef.current.rotation.z = headRoll;
  });

  return (
    <group position={[0, 0, 0]}>
      <group ref={headRef}>
        <mesh>
          <sphereGeometry args={[1.35, 64, 64]} />
          <meshStandardMaterial color="#d8a27f" roughness={0.65} />
        </mesh>

        <mesh position={[-0.48, 0.28, 1.18]}>
          <sphereGeometry args={[0.14, 32, 32]} />
          <meshStandardMaterial color="#ffffff" />
        </mesh>

        <mesh position={[0.48, 0.28, 1.18]}>
          <sphereGeometry args={[0.14, 32, 32]} />
          <meshStandardMaterial color="#ffffff" />
        </mesh>

        <mesh position={[-0.48, 0.28, 1.31]}>
          <sphereGeometry args={[0.06, 32, 32]} />
          <meshStandardMaterial color="#151515" />
        </mesh>

        <mesh position={[0.48, 0.28, 1.31]}>
          <sphereGeometry args={[0.06, 32, 32]} />
          <meshStandardMaterial color="#151515" />
        </mesh>

        <mesh
          position={[
            -0.48,
            0.62 + browRaise * 0.12,
            1.25,
          ]}
          rotation={[0, 0, -0.1]}
        >
          <boxGeometry args={[0.38, 0.055, 0.06]} />
          <meshStandardMaterial color="#3c2c25" />
        </mesh>

        <mesh
          position={[
            0.48,
            0.62 + browRaise * 0.12,
            1.25,
          ]}
          rotation={[0, 0, 0.1]}
        >
          <boxGeometry args={[0.38, 0.055, 0.06]} />
          <meshStandardMaterial color="#3c2c25" />
        </mesh>

        <mesh position={[0, -0.42, 1.27]}>
          <boxGeometry
            args={[
              0.48 + mouthWidth * 0.38,
              0.06 + mouthOpen * 0.32,
              0.08,
            ]}
          />
          <meshStandardMaterial color="#7b2020" />
        </mesh>

        <mesh position={[0, 0.02, 1.28]}>
          <coneGeometry args={[0.12, 0.35, 32]} />
          <meshStandardMaterial color="#c68f70" />
        </mesh>
      </group>

      <mesh position={[0, -1.9, 0]}>
        <cylinderGeometry args={[0.52, 0.7, 1.2, 32]} />
        <meshStandardMaterial color="#d8a27f" />
      </mesh>

      <mesh position={[0, -2.85, 0]}>
        <boxGeometry args={[2.8, 1.5, 1.1]} />
        <meshStandardMaterial color="#202938" />
      </mesh>
    </group>
  );
}

export default DigitalHuman;