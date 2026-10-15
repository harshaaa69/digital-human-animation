
import { useMemo, useEffect } from "react";
import { useFrame, useLoader, useThree } from "@react-three/fiber";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import type { Mesh, Object3D } from "three";

type AvatarModelProps = {
  frame: number[];
};

function AvatarModel({ frame }: AvatarModelProps) {
  const gl = useThree((state) => state.gl);

  const ktx2Loader = useMemo(() => {
    return new KTX2Loader()
      .setTranscoderPath("/basis/")
      .detectSupport(gl);
  }, [gl]);

  const gltf = useLoader(
    GLTFLoader,
    "/models/avatar.glb",
    (loader) => {
      loader.setKTX2Loader(ktx2Loader);
      loader.setMeshoptDecoder(MeshoptDecoder);
    }
  );

  const scene = gltf.scene;

  useEffect(() => {
    scene.traverse((object: Object3D) => {
      const mesh = object as Mesh;

      if (
        mesh.morphTargetDictionary &&
        mesh.morphTargetInfluences
      ) {
        console.log(
          "Avatar mesh:",
          mesh.name,
          mesh.morphTargetDictionary
        );
      }
    });
  }, [scene]);

  useFrame(() => {
    const mouthOpen = frame[0] ?? 0;
    const mouthWidth = frame[1] ?? 0;
    const browRaise = frame[2] ?? 0;
    const headYaw = frame[3] ?? 0;
    const headPitch = frame[4] ?? 0;
    const headRoll = frame[5] ?? 0;

    scene.rotation.set(
      headPitch * 0.5,
      headYaw * 0.5,
      headRoll * 0.5
    );

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
        Object.entries(dictionary).forEach(
          ([name, index]) => {
            const normalizedName = name
              .split(".")
              .pop()
              ?.toLowerCase();

            if (
              normalizedName &&
              names.some(
                (target) =>
                  normalizedName === target.toLowerCase()
              )
            ) {
              influences[index] = Math.max(
                0,
                Math.min(1, value)
              );
            }
          }
        );
      };

      setMorph(
        ["jawOpen", "mouthOpen"],
        mouthOpen
      );

      setMorph(
        ["mouthStretchLeft", "mouthStretchRight"],
        mouthWidth * 0.4
      );

      setMorph(
        [
          "browInnerUp",
          "browOuterUpLeft",
          "browOuterUpRight"
        ],
        browRaise * 0.6
      );
    });
  });

  return <primitive object={scene} />;
}

export default AvatarModel;
