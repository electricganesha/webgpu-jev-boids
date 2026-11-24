import * as THREE from 'three/webgpu';
import type { StorageBufferNode } from 'three/webgpu';
import {
  Fn,
  attribute,
  varying,
  instanceIndex,
  vec2,
  vec3,
  vec4,
  float,
  mix,
  clamp,
  dot,
  normalize,
  select,
  pow,
  abs,
  smoothstep,
  length,
  hash,
} from 'three/tsl';
import type { BoidSimUniformNodes } from './boidsComputeTSL';

export function createBoidGeometry(): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();

  // 12 vertices (4 triangles: 2 shadow, 2 body)
  const dirFactors = new Float32Array([
    // Triangle 0: Left wing shadow (Tip, Left wing, Spine)
    1.0, -0.72, -0.2736,
    // Triangle 1: Right wing shadow (Tip, Spine, Right wing)
    1.0, -0.2736, -0.72,
    // Triangle 2: Left wing body (Tip, Left wing, Spine)
    1.0, -0.72, -0.2736,
    // Triangle 3: Right wing body (Tip, Spine, Right wing)
    1.0, -0.2736, -0.72,
  ]);

  const perpFactors = new Float32Array([
    // Tri 0
    0.0, -1.0, 0.0,
    // Tri 1
    0.0, 0.0, 1.0,
    // Tri 2
    0.0, -1.0, 0.0,
    // Tri 3
    0.0, 0.0, 1.0,
  ]);

  const sides = new Float32Array([
    // Tri 0
    0.0, -1.0, 0.0,
    // Tri 1
    0.0, 0.0, 1.0,
    // Tri 2
    0.0, -1.0, 0.0,
    // Tri 3
    0.0, 0.0, 1.0,
  ]);

  const isShadows = new Float32Array([
    // Shadow triangles (0..5)
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    // Body triangles (6..11)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  ]);

  const isLeftWings = new Float32Array([
    // Tri 0 (Left)
    1.0, 1.0, 1.0,
    // Tri 1 (Right)
    0.0, 0.0, 0.0,
    // Tri 2 (Left)
    1.0, 1.0, 1.0,
    // Tri 3 (Right)
    0.0, 0.0, 0.0,
  ]);

  const uvs = new Float32Array([
    // Tri 0
    0.5, 1.0, 0.0, 0.0, 0.5, 0.25,
    // Tri 1
    0.5, 1.0, 0.5, 0.25, 1.0, 0.0,
    // Tri 2
    0.5, 1.0, 0.0, 0.0, 0.5, 0.25,
    // Tri 3
    0.5, 1.0, 0.5, 0.25, 1.0, 0.0,
  ]);

  const dummyPositions = new Float32Array(12 * 3); // 36 floats, all 0

  geometry.setAttribute('position', new THREE.BufferAttribute(dummyPositions, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setAttribute('aDirFactor', new THREE.BufferAttribute(dirFactors, 1));
  geometry.setAttribute('aPerpFactor', new THREE.BufferAttribute(perpFactors, 1));
  geometry.setAttribute('aSide', new THREE.BufferAttribute(sides, 1));
  geometry.setAttribute('aIsShadow', new THREE.BufferAttribute(isShadows, 1));
  geometry.setAttribute('aIsLeftWing', new THREE.BufferAttribute(isLeftWings, 1));

  return geometry;
}

export function createBoidsNodeMaterial(
  particleStorage: StorageBufferNode<"vec4">,
  u: BoidSimUniformNodes
): THREE.MeshBasicNodeMaterial {
  const material = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  });

  const aDirFactor = attribute('aDirFactor', 'float');
  const aPerpFactor = attribute('aPerpFactor', 'float');
  const aSide = attribute('aSide', 'float');
  const aIsShadow = attribute('aIsShadow', 'float');
  const aIsLeftWing = attribute('aIsLeftWing', 'float');
  const aUv = attribute('uv', 'vec2');

  // Varying definitions passed from vertex to fragment stage
  const vUv = varying(vec2(0.0, 0.0));
  const vDiffuse = varying(float(0.0));
  const vSide = varying(float(0.0));
  const vIsShadow = varying(float(0.0));
  const vBoidHash = varying(float(0.0));

  material.positionNode = Fn(() => {
    const p = particleStorage.element(instanceIndex);
    const pos = p.xy;
    const vel = p.zw;
    const speed = length(vel);

    const minS = clamp(u.minSpeed, float(0.05), float(1.0));
    const maxS = clamp(u.maxSpeed, minS.add(0.1), float(2.0));
    const speedNorm = clamp(speed.sub(minS).div(maxS.sub(minS)), float(0.0), float(1.0));

    // Per-boid hash for natural individual variance
    const bHash = hash(instanceIndex);
    const depthScale = mix(float(0.80), float(1.25), bHash);

    // Heading direction
    const dir = select(speed.greaterThan(0.0001), vel.div(speed), vec2(0.0, 1.0));
    const perp = vec2(dir.y.negate(), dir.x);

    // Individual dimensions
    const lengthScale = float(0.0105).add(speedNorm.mul(0.005)).mul(depthScale);
    const widthScale = float(0.0044).add(speedNorm.mul(0.0014)).mul(depthScale);

    const localPos = dir.mul(aDirFactor.mul(lengthScale)).add(perp.mul(aPerpFactor.mul(widthScale)));

    // Simulated 3D surface normal for distinct left/right origami wing facet lighting
    const normalLeft = normalize(
      vec3(perp.x.negate().mul(0.75).add(dir.x.mul(0.15)), perp.y.negate().mul(0.75).add(dir.y.mul(0.15)), float(0.65))
    );
    const normalRight = normalize(
      vec3(perp.x.mul(0.75).add(dir.x.mul(0.15)), perp.y.mul(0.75).add(dir.y.mul(0.15)), float(0.65))
    );
    const normal3d = select(aIsLeftWing.greaterThan(0.5), normalLeft, normalRight);

    // Sun direction from top-left
    const sunDir = normalize(vec3(-0.45, 0.65, 0.60));
    const diffuse = clamp(dot(normal3d, sunDir).mul(0.55).add(0.55), float(0.25), float(1.15));

    const shadowOffset = select(
      aIsShadow.greaterThan(0.5),
      vec2(depthScale.mul(-0.003), depthScale.mul(-0.004)),
      vec2(0.0, 0.0)
    );

    const worldPos = pos.add(localPos).add(shadowOffset);

    // Pass varyings to fragment stage
    vUv.assign(aUv);
    vDiffuse.assign(diffuse);
    vSide.assign(aSide);
    vIsShadow.assign(aIsShadow);
    vBoidHash.assign(bHash);

    // Clip position mapped by aspect ratio
    return vec3(worldPos.x.div(u.aspectRatio), worldPos.y, float(0.0));
  })();

  material.colorNode = Fn(() => {
    // 1. Diluted Ink Bleed Drop Shadow
    const dCenter = abs(vUv.x.sub(0.5)).mul(2.0);
    const washFade = clamp(float(1.0).sub(dCenter.mul(0.65)), float(0.0), float(1.0));
    const shadowColor = vec4(vec3(0.22, 0.24, 0.28), washFade.mul(0.12));

    // 2. Individualized Ink Tone Spectrum
    const lightWash = vec3(0.48, 0.51, 0.56);
    const mediumInk = vec3(0.22, 0.24, 0.28);
    const denseCarbon = vec3(0.04, 0.045, 0.055);

    const h = vBoidHash;
    const boidInk = select(
      h.lessThan(0.45),
      mix(lightWash, mediumInk, h.div(0.45)),
      mix(mediumInk, denseCarbon, h.sub(0.45).div(0.55))
    );

    // 3. Dynamic Wing Facet Contrast
    const facetFactor = mix(float(0.48), float(1.25), vDiffuse);
    const litInk = boidInk.mul(facetFactor).toVar();

    // 4. Concentrated Ink Spine / Keel Crease
    const spineProximity = float(1.0).sub(abs(vSide));
    const spineRidge = pow(clamp(spineProximity, float(0.0), float(1.0)), float(3.0));
    litInk.assign(mix(litInk, denseCarbon.mul(0.8), spineRidge.mul(0.35)));

    // Wet ink sheen highlight
    const wetSheen = pow(clamp(spineProximity, float(0.0), float(1.0)), float(5.0))
      .mul(vDiffuse)
      .mul(0.18);
    litInk.addAssign(vec3(wetSheen));

    // 5. Crisp Outer Demarcation / Micro Rim Contour
    const edgeProximity = abs(vSide);
    const outerRim = smoothstep(float(0.85), float(0.98), edgeProximity);
    litInk.assign(mix(litInk, denseCarbon.mul(1.1), outerRim.mul(0.3)));

    const finalRgb = clamp(litInk, vec3(0.0), vec3(1.0));

    // Opacity
    const baseAlpha = mix(float(0.72), float(0.96), h);
    const alpha = spineProximity.mul(0.3).add(0.7).mul(baseAlpha);

    const bodyColor = vec4(finalRgb, alpha);

    return select(vIsShadow.greaterThan(0.5), shadowColor, bodyColor);
  })();

  return material;
}
