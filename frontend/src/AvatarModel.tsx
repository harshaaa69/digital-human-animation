import { useEffect, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import type {
  Group,
  Mesh,
  Object3D,
} from "three";

type AvatarModelProps = {
  frame: number[];
};

function AvatarModel({ frame }: AvatarModelProps) {
  const groupRef = useRef<Group>(null);

  const { scene } = useGLTF("/models/avatar.glb");

  const mouthOpen = frame[0] ?? 0;
  const mouthWidth = frame[1] ?? 0;
  const browRaise = frame[2] ?? 0;
  const headYaw = frame[3] ?? 0;
  const headPitch = frame[4] ?? 0;
  const headRoll = frame[5] ?? 0;

  useEffect(() => {
    scene.traverse((object: Object3D) => {
      const mesh = object as Mesh;

      if (
        mesh.morphTargetDictionary &&
        mesh.morphTargetInfluences
      ) {
        console.log(
          `Morph targets for ${mesh.name}`,
          mesh.morphTargetDictionary
        );
      }
    });
  }, [scene]);

  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.rotation.y = headYaw;
      groupRef.current.rotation.x = headPitch;
      groupRef.current.rotation.z = headRoll;
    }

    scene.traverse((object: Object3D) => {
      const mesh = object as Mesh;

      if (
        !mesh.morphTargetDictionary ||
        !mesh.morphTargetInfluences
      ) {
        return;
      }

      const dictionary = mesh.morphTargetDictionary;
      const influences = mesh.morphTargetInfluences;

      const setMorph = (
        names: string[],
        value: number
      ) => {
        for (const name of names) {
          const index = dictionary[name];

          if (index !== undefined) {
            influences[index] = value;
          }
        }
      };

      setMorph(
        [
          "jawOpen",
          "Jaw_Open",
          "mouthOpen",
          "Mouth_Open",
          "viseme_aa",
        ],
        mouthOpen
      );

      setMorph(
        [
          "mouthSmileLeft",
          "mouthSmileRight",
          "Mouth_Smile",
        ],
        mouthWidth * 0.35
      );

      setMorph(
        [
          "browInnerUp",
          "browOuterUpLeft",
          "browOuterUpRight",
          "Brow_Raise",
        ],
        browRaise
      );
    });
  });

  return (
    <group
      ref={groupRef}
      position={[0, -2.6, 0]}
      scale={2}
    >
      <primitive object={scene} />
    </group>
  );
}

useGLTF.preload("/models/avatar.glb");

export default AvatarModel;