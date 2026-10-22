
import { useEffect, useMemo, useRef } from "react";
import { useFrame, useLoader, useThree } from "@react-three/fiber";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { MathUtils, Mesh, Object3D } from "three";

type AvatarModelProps = {
  frame: number[];
};

type MorphMesh = Mesh & {
  morphTargetDictionary: Record<string, number>;
  morphTargetInfluences: number[];
};

const MORPH_ALIASES: Record<string, string> = {
  browouterupleft: "browouterupleft",
  browouterupright: "browouterupright",
  browouterupl: "browouterupleft",
  browouterupr: "browouterupright",
  mouthstretchl: "mouthstretchleft",
  mouthstretchr: "mouthstretchright",
  mouthstretchleft: "mouthstretchleft",
  mouthstretchright: "mouthstretchright",
  mouthpressl: "mouthpressleft",
  mouthpressr: "mouthpressright",
  mouthpressleft: "mouthpressleft",
  mouthpressright: "mouthpressright",
  mouthsmilel: "mouthsmileleft",
  mouthsmiler: "mouthsmileright",
  mouthsmileleft: "mouthsmileleft",
  mouthsmileright: "mouthsmileright",
  mouthlowerdownl: "mouthlowerdownleft",
  mouthlowerdownr: "mouthlowerdownright",
  mouthlowerdownleft: "mouthlowerdownleft",
  mouthlowerdownright: "mouthlowerdownright",
};

function clamp(value: number): number {
  return MathUtils.clamp(
    Number.isFinite(value) ? value : 0,
    0,
    1
  );
}

function normalizeName(name: string): string {
  const normalized =
    name
      .split(".")
      .pop()
      ?.replace(/[^a-zA-Z0-9]/g, "")
      .toLowerCase() ?? "";

  return MORPH_ALIASES[normalized] ?? normalized;
}

function isMorphMesh(object: Object3D): object is MorphMesh {
  const mesh = object as Mesh;

  return Boolean(
    mesh.isMesh &&
      mesh.morphTargetDictionary &&
      mesh.morphTargetInfluences
  );
}

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

  const meshes = useRef<MorphMesh[]>([]);
  const currentFrame = useRef<number[]>(
    Array(11).fill(0)
  );

  useEffect(() => {
    const found: MorphMesh[] = [];

    scene.traverse((object: Object3D) => {
      if (isMorphMesh(object)) {
        found.push(object);
      }
    });

    meshes.current = found;

    return () => {
      meshes.current = [];
    };
  }, [scene]);

  useFrame((_, delta) => {
    const alpha = 1 - Math.exp(
      -Math.min(delta, 0.1) * 18
    );

    const current = currentFrame.current;

    for (let index = 0; index < 11; index++) {
      const target = clamp(frame[index] ?? 0);

      current[index] = MathUtils.lerp(
        current[index],
        target,
        alpha
      );
    }

    const mouthOpen = current[0];
    const mouthWidth = current[1];
    const browRaise = current[2];
    const headYaw = current[3];
    const headPitch = current[4];
    const headRoll = current[5];

    const hasFacialChannels = frame.length >= 11;

    const lipClosure = hasFacialChannels
      ? current[7]
      : clamp((1 - mouthOpen) * 0.15);

    const lipRounding = hasFacialChannels
      ? current[8]
      : clamp((1 - mouthWidth) * mouthOpen * 0.5);

    const lipStretch = hasFacialChannels
      ? current[9]
      : clamp(mouthWidth * 0.4);

    const lowerLipContact = hasFacialChannels
      ? current[10]
      : 0;

    const effectiveJawOpen = clamp(
      mouthOpen * (1 - lipClosure * 0.8)
    );

    scene.rotation.set(
      headPitch * 0.5,
      headYaw * 0.5,
      headRoll * 0.5
    );

    const values: Record<string, number> = {
      jawopen: effectiveJawOpen,

      mouthclose: clamp(lipClosure),

      mouthpressleft: clamp(
        lipClosure * 0.48
      ),

      mouthpressright: clamp(
        lipClosure * 0.48
      ),

      mouthfunnel: clamp(
        lipRounding * 0.75
      ),

      mouthpucker: clamp(
        lipRounding * 0.55
      ),

      mouthstretchleft: clamp(
        lipStretch * 0.65
      ),

      mouthstretchright: clamp(
        lipStretch * 0.65
      ),

      mouthsmileleft: clamp(
        Math.max(0, mouthWidth - 0.6) * 0.2
      ),

      mouthsmileright: clamp(
        Math.max(0, mouthWidth - 0.6) * 0.2
      ),

      mouthrolllower: clamp(
        lowerLipContact * 0.5
      ),

      mouthshruglower: clamp(
        lowerLipContact * 0.35
      ),

      mouthlowerdownleft: clamp(
        mouthOpen * 0.22
      ),

      mouthlowerdownright: clamp(
        mouthOpen * 0.22
      ),

      browinnerup: clamp(
        browRaise * 0.6
      ),

      browouterupleft: clamp(
        browRaise * 0.6
      ),

      browouterupright: clamp(
        browRaise * 0.6
      ),
    };

    for (const mesh of meshes.current) {
      for (const [name, index] of Object.entries(
        mesh.morphTargetDictionary
      )) {
        const normalized = normalizeName(name);
        const value = values[normalized];

        if (value !== undefined) {
          mesh.morphTargetInfluences[index] = clamp(value);
        }
      }
    }
  });

  return <primitive object={scene} />;
}

export default AvatarModel;
