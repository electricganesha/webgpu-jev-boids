import type { StorageBufferNode, ComputeNode } from 'three/webgpu';
import {
  Fn,
  Loop,
  If,
  float,
  uint,
  vec2,
  vec4,
  instanceIndex,
  localId,
  workgroupId,
  workgroupArray,
  workgroupBarrier,
  clamp,
  length,
  min,
  max,
  smoothstep,
  uniform,
} from 'three/tsl';

type FloatUniform = ReturnType<typeof uniform<'float'>>;
type Vec2Uniform = ReturnType<typeof uniform<'vec2'>>;

export interface BoidSimUniformNodes {
  deltaTime: FloatUniform;
  numParticles: FloatUniform;
  cohesionWeight: FloatUniform;
  separationWeight: FloatUniform;
  alignmentWeight: FloatUniform;
  maxSpeed: FloatUniform;
  minSpeed: FloatUniform;
  targetAttractionWeight: FloatUniform;
  aspectRatio: FloatUniform;
  targetPos: Vec2Uniform;
  centroid: Vec2Uniform;
  vortexWeight: FloatUniform;
  repelRadius: FloatUniform;
  alignRadius: FloatUniform;
  cohesionRadius: FloatUniform;
  mouseActive: FloatUniform;
}

const WORKGROUP_SIZE = 64;

export function createBoidsComputeNode(
  storageIn: StorageBufferNode<"vec4">,
  storageOut: StorageBufferNode<"vec4">,
  u: BoidSimUniformNodes,
  totalParticles: number
): ComputeNode {
  const computeFn = Fn(() => {
    const idx = instanceIndex;
    const aspect = max(u.aspectRatio, float(0.1));
    const totalCount = max(u.numParticles, float(64.0)).toUint();
    const totalGroups = max(float(1.0), u.numParticles.div(float(WORKGROUP_SIZE))).toUint();

    const myParticle = storageIn.element(idx);
    const pos = myParticle.xy.toVar();
    const vel = myParticle.zw.toVar();

    const separation = vec2(0.0, 0.0).toVar();
    const alignment = vec2(0.0, 0.0).toVar();
    const cohesionCenter = vec2(0.0, 0.0).toVar();
    const alignCount = float(0.0).toVar();
    const cohesionCount = float(0.0).toVar();

    const rSep = max(u.repelRadius, float(0.01));
    const rAlign = max(u.alignRadius, float(0.02));
    const rCoh = max(u.cohesionRadius, float(0.04));

    // Shared tile memory: 64 particles per workgroup
    const tile = workgroupArray('vec4', WORKGROUP_SIZE) as any;

    // Sample 2 tiles (128 neighbors) cooperatively
    // Tile 0: Local workgroup
    // Tile 1: Strided pseudo-random block across the buffer
    Loop(2, ({ i: t }) => {
      const tileGroup = workgroupId.x.add(t.toUint().mul(uint(47))).mod(totalGroups);
      const sampleIdx = tileGroup.mul(uint(WORKGROUP_SIZE)).add(localId.x);

      If(sampleIdx.lessThan(totalCount), () => {
        tile.element(localId.x).assign(storageIn.element(sampleIdx));
      }).Else(() => {
        tile.element(localId.x).assign(myParticle);
      });

      workgroupBarrier();

      Loop(WORKGROUP_SIZE, ({ i: j }) => {
        const other = tile.element(j);
        const otherPos = other.xy;
        const otherVel = other.zw;
        const diff = pos.sub(otherPos);
        const dist = length(diff);

        If(dist.greaterThan(0.0001), () => {
          // Separation
          If(dist.lessThan(rSep), () => {
            const rep = diff.div(dist.mul(dist).add(0.0015));
            separation.addAssign(rep);
          });

          // Alignment
          If(dist.lessThan(rAlign), () => {
            alignment.addAssign(otherVel);
            alignCount.addAssign(1.0);
          });

          // Cohesion
          If(dist.lessThan(rCoh), () => {
            cohesionCenter.addAssign(otherPos);
            cohesionCount.addAssign(1.0);
          });
        });
      });

      workgroupBarrier();
    });

    // Compute net acceleration
    const accel = vec2(0.0, 0.0).toVar();

    // 1. Clamped Separation
    const sepLen = length(separation);
    If(sepLen.greaterThan(12.0), () => {
      separation.assign(separation.div(sepLen).mul(12.0));
    });
    accel.addAssign(separation.mul(u.separationWeight));

    // 2. Alignment
    If(alignCount.greaterThan(0.0), () => {
      const avgVel = alignment.div(alignCount);
      accel.addAssign(avgVel.sub(vel).mul(u.alignmentWeight));
    });

    // 3. Cohesion
    If(cohesionCount.greaterThan(0.0), () => {
      const center = cohesionCenter.div(cohesionCount);
      const cohDir = center.sub(pos);
      const cohDist = length(cohDir);
      If(cohDist.greaterThan(0.001), () => {
        accel.addAssign(
          cohDir
            .div(cohDist)
            .mul(min(cohDist.mul(4.0), float(2.0)))
            .mul(u.cohesionWeight)
        );
      });
    });

    // 4. Cursor Attractor / Repellor (localized threat envelope)
    If(u.mouseActive.greaterThan(0.5), () => {
      const targetDiff = u.targetPos.sub(pos);
      const targetDist = length(targetDiff);
      If(targetDist.greaterThan(0.001), () => {
        const dir = targetDiff.div(targetDist);
        If(u.targetAttractionWeight.lessThan(0.0), () => {
          // Evasion bubble with smooth hermite falloff
          const threatRadius = float(0.38);
          If(targetDist.lessThan(threatRadius), () => {
            const factor = smoothstep(threatRadius, float(0.02), targetDist);
            accel.addAssign(dir.mul(u.targetAttractionWeight.mul(factor).mul(float(5.0))));
          });
        }).Else(() => {
          // Attractor pull
          const falloff = float(1.0).div(float(1.0).add(targetDist.mul(2.5)));
          accel.addAssign(dir.mul(u.targetAttractionWeight.mul(falloff).mul(float(4.0))));
        });
      });
    });

    // 5. Vortex Coil force (stable orbital swirling around current swarm centroid)
    If(u.vortexWeight.greaterThan(0.01), () => {
      const rVec = pos.sub(u.centroid);
      const rDist = length(rVec);
      If(rDist.greaterThan(0.001), () => {
        const tangent = vec2(rVec.y.negate(), rVec.x).div(rDist);
        const desiredRadius = float(0.44);
        const radialSteer = rVec.div(rDist).mul(desiredRadius.sub(rDist)).mul(float(2.2));
        accel.addAssign(
          tangent
            .mul(float(3.5))
            .add(radialSteer)
            .mul(u.vortexWeight)
        );
      });
    });

    // 6. Soft boundary bounce
    const boundX = aspect.sub(0.14);
    const boundY = float(0.85);
    const wallK = float(7.5);

    If(pos.x.greaterThan(boundX), () => {
      accel.x.subAssign(pos.x.sub(boundX).mul(wallK));
    }).Else(() => {
      If(pos.x.lessThan(boundX.negate()), () => {
        accel.x.subAssign(pos.x.add(boundX).mul(wallK));
      });
    });

    If(pos.y.greaterThan(boundY), () => {
      accel.y.subAssign(pos.y.sub(boundY).mul(wallK));
    }).Else(() => {
      If(pos.y.lessThan(boundY.negate()), () => {
        accel.y.subAssign(pos.y.add(boundY).mul(wallK));
      });
    });

    // 7. Integrate acceleration into velocity
    const dt = clamp(u.deltaTime, float(0.001), float(0.033));
    vel.addAssign(accel.mul(dt));

    // 8. Speed clamping
    const speed = length(vel);
    const minSpd = max(u.minSpeed, float(0.1));
    const maxSpd = max(u.maxSpeed, float(0.3));

    If(speed.greaterThan(maxSpd), () => {
      vel.assign(vel.div(speed).mul(maxSpd));
    }).Else(() => {
      If(speed.lessThan(minSpd), () => {
        If(speed.greaterThan(0.0001), () => {
          vel.assign(vel.div(speed).mul(minSpd));
        }).Else(() => {
          vel.assign(vec2(minSpd, float(0.0)));
        });
      });
    });

    // 9. Integrate position
    pos.addAssign(vel.mul(dt));

    // 10. Write updated state to output buffer
    storageOut.element(idx).assign(vec4(pos.x, pos.y, vel.x, vel.y));
  });

  return computeFn().compute(totalParticles, [WORKGROUP_SIZE, 1, 1]);
}
