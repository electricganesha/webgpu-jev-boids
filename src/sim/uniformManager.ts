import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import type { InstinctProfile, InstinctType, SimUniforms } from './types';
import type { BoidSimUniformNodes } from '../shaders/boidsComputeTSL';

export const INSTINCT_PROFILES: Record<InstinctType, InstinctProfile> = {
  murmuration: {
    label: 'Murmuration',
    description: 'Calm, expansive sweeping arcs; high cohesion and smooth alignment.',
    cohesion_weight: 1.2,
    separation_weight: 1.4,
    alignment_weight: 1.8,
    max_speed: 0.65,
    min_speed: 0.25,
    target_attraction_weight: 0.4,
    vortex_weight: 0.0,
    repel_radius: 0.035,
    align_radius: 0.12,
    cohesion_radius: 0.22,
    color_theme: 0,
  },
  scatter_panic: {
    label: 'Evasive Murmuration',
    description: 'Cohesive evasive banking; high alignment and synchronized flight away from threat.',
    cohesion_weight: 1.1,
    separation_weight: 1.8,
    alignment_weight: 2.2,
    max_speed: 1.15,
    min_speed: 0.35,
    target_attraction_weight: -2.2,
    vortex_weight: 0.0,
    repel_radius: 0.045,
    align_radius: 0.14,
    cohesion_radius: 0.24,
    color_theme: 1,
  },
  vortex_coil: {
    label: 'Vortex Coil',
    description: 'Graceful orbital swirling around centroid; organic circular flight.',
    cohesion_weight: 1.6,
    separation_weight: 1.6,
    alignment_weight: 1.8,
    max_speed: 0.85,
    min_speed: 0.35,
    target_attraction_weight: 0.15,
    vortex_weight: 1.5,
    repel_radius: 0.04,
    align_radius: 0.18,
    cohesion_radius: 0.35,
    color_theme: 2,
  },
  predator_mob: {
    label: 'Predator Mob',
    description: 'Aggressive convergence directly toward the cursor target.',
    cohesion_weight: 2.2,
    separation_weight: 1.1,
    alignment_weight: 1.5,
    max_speed: 1.35,
    min_speed: 0.45,
    target_attraction_weight: 2.5,
    vortex_weight: 0.0,
    repel_radius: 0.03,
    align_radius: 0.14,
    cohesion_radius: 0.28,
    color_theme: 3,
  },
};

export class UniformManager {
  public current: SimUniforms;
  public target: SimUniforms;
  public activeInstinct: InstinctType = 'murmuration';
  public currentConfidence: number = 0.9;
  public invertPolarity: boolean = false;
  public polarityMode: 'attractor' | 'repulsor' = 'attractor';

  private bufferArray: ArrayBuffer;
  private floatView: Float32Array;
  private uintView: Uint32Array;

  constructor(numParticles: number = 64000, aspectRatio: number = 1.0) {
    const initialProfile = INSTINCT_PROFILES.murmuration;

    this.current = {
      delta_time: 0.016,
      num_particles: numParticles,
      cohesion_weight: initialProfile.cohesion_weight,
      separation_weight: initialProfile.separation_weight,
      alignment_weight: initialProfile.alignment_weight,
      max_speed: initialProfile.max_speed,
      target_attraction_weight: initialProfile.target_attraction_weight,
      aspect_ratio: aspectRatio,
      target_pos: [0.0, 0.0],
      centroid: [0.0, 0.0],
      vortex_weight: initialProfile.vortex_weight,
      repel_radius: initialProfile.repel_radius,
      align_radius: initialProfile.align_radius,
      cohesion_radius: initialProfile.cohesion_radius,
      color_theme: initialProfile.color_theme,
      min_speed: initialProfile.min_speed,
      mouse_active: 0.0,
    };

    this.target = { ...this.current };

    // 80 bytes (20 floats/uints of 4 bytes each = 80 bytes, divisible by 16)
    this.bufferArray = new ArrayBuffer(80);
    this.floatView = new Float32Array(this.bufferArray);
    this.uintView = new Uint32Array(this.bufferArray);
  }

  public setPolarityMode(mode: 'attractor' | 'repulsor'): void {
    this.polarityMode = mode;
    this.invertPolarity = mode === 'repulsor';
    const profile = INSTINCT_PROFILES[this.activeInstinct];
    let attraction = profile.target_attraction_weight;
    if (this.polarityMode === 'repulsor') {
      attraction = -Math.max(2.2, Math.abs(attraction));
    } else {
      attraction = Math.max(1.8, Math.abs(attraction));
    }
    this.target.target_attraction_weight = attraction;
    this.current.target_attraction_weight = attraction;
  }

  public setInstinct(instinct: InstinctType, confidence: number, invertPolarity: boolean = false): void {
    this.activeInstinct = instinct;
    this.currentConfidence = Math.max(0.1, Math.min(1.0, confidence));

    const profile = INSTINCT_PROFILES[instinct];
    let attraction = profile.target_attraction_weight;
    if (this.polarityMode === 'repulsor') {
      attraction = -Math.max(2.2, Math.abs(attraction));
      this.invertPolarity = true;
    } else if (this.polarityMode === 'attractor') {
      attraction = Math.max(1.8, Math.abs(attraction));
      this.invertPolarity = false;
    } else if (invertPolarity) {
      attraction = -Math.abs(attraction || 2.0);
      this.invertPolarity = true;
    } else {
      this.invertPolarity = false;
    }

    this.target.cohesion_weight = profile.cohesion_weight;
    this.target.separation_weight = profile.separation_weight;
    this.target.alignment_weight = profile.alignment_weight;
    this.target.max_speed = profile.max_speed;
    this.target.min_speed = profile.min_speed;
    this.target.target_attraction_weight = attraction;
    this.target.vortex_weight = profile.vortex_weight;
    this.target.repel_radius = profile.repel_radius;
    this.target.align_radius = profile.align_radius;
    this.target.cohesion_radius = profile.cohesion_radius;
    this.target.color_theme = profile.color_theme;
  }

  public updateTargetPos(x: number, y: number, active: boolean): void {
    this.target.target_pos = [x, y];
    this.target.mouse_active = active ? 1.0 : 0.0;
  }

  public updateCentroid(centroid: [number, number]): void {
    this.target.centroid = centroid;
    // Fast blend centroid so vortex centers accurately
    this.current.centroid[0] += (centroid[0] - this.current.centroid[0]) * 0.1;
    this.current.centroid[1] += (centroid[1] - this.current.centroid[1]) * 0.1;
  }

  public setAspectRatio(aspect: number): void {
    this.current.aspect_ratio = aspect;
    this.target.aspect_ratio = aspect;
  }

  public setParticleCount(count: number): void {
    this.current.num_particles = count;
    this.target.num_particles = count;
  }

  public update(dt: number): void {
    this.current.delta_time = dt;

    // Modulate LERP speed based on calibrated confidence
    // Smooth, organic glide that breathes naturally (rate ~ 1.8 to 3.8)
    const lerpRate = 1.8 + this.currentConfidence * 2.0;

    const factor = Math.min(1.0, 1.0 - Math.exp(-lerpRate * dt));

    const lerp = (cur: number, tgt: number) => cur + (tgt - cur) * factor;

    this.current.cohesion_weight = lerp(this.current.cohesion_weight, this.target.cohesion_weight);
    this.current.separation_weight = lerp(this.current.separation_weight, this.target.separation_weight);
    this.current.alignment_weight = lerp(this.current.alignment_weight, this.target.alignment_weight);
    this.current.max_speed = lerp(this.current.max_speed, this.target.max_speed);
    this.current.min_speed = lerp(this.current.min_speed, this.target.min_speed);
    this.current.target_attraction_weight = lerp(this.current.target_attraction_weight, this.target.target_attraction_weight);
    this.current.vortex_weight = lerp(this.current.vortex_weight, this.target.vortex_weight);
    this.current.repel_radius = lerp(this.current.repel_radius, this.target.repel_radius);
    this.current.align_radius = lerp(this.current.align_radius, this.target.align_radius);
    this.current.cohesion_radius = lerp(this.current.cohesion_radius, this.target.cohesion_radius);

    // Mouse positions snap smoothly
    this.current.target_pos[0] = lerp(this.current.target_pos[0], this.target.target_pos[0]);
    this.current.target_pos[1] = lerp(this.current.target_pos[1], this.target.target_pos[1]);
    this.current.mouse_active = this.target.mouse_active;

    // Theme switches once past midpoint
    this.current.color_theme = this.target.color_theme;
  }

  public getRawBuffer(): ArrayBuffer {
    // Pack 80 bytes strictly according to WGSL SimUniforms layout:
    // 0: delta_time (f32)
    this.floatView[0] = this.current.delta_time;
    // 1: num_particles (u32)
    this.uintView[1] = this.current.num_particles;
    // 2: cohesion_weight (f32)
    this.floatView[2] = this.current.cohesion_weight;
    // 3: separation_weight (f32)
    this.floatView[3] = this.current.separation_weight;
    // 4: alignment_weight (f32)
    this.floatView[4] = this.current.alignment_weight;
    // 5: max_speed (f32)
    this.floatView[5] = this.current.max_speed;
    // 6: target_attraction_weight (f32)
    this.floatView[6] = this.current.target_attraction_weight;
    // 7: aspect_ratio (f32)
    this.floatView[7] = this.current.aspect_ratio;
    // 8, 9: target_pos (vec2<f32>)
    this.floatView[8] = this.current.target_pos[0];
    this.floatView[9] = this.current.target_pos[1];
    // 10, 11: centroid (vec2<f32>)
    this.floatView[10] = this.current.centroid[0];
    this.floatView[11] = this.current.centroid[1];
    // 12: vortex_weight (f32)
    this.floatView[12] = this.current.vortex_weight;
    // 13: repel_radius (f32)
    this.floatView[13] = this.current.repel_radius;
    // 14: align_radius (f32)
    this.floatView[14] = this.current.align_radius;
    // 15: cohesion_radius (f32)
    this.floatView[15] = this.current.cohesion_radius;
    // 16: color_theme (u32)
    this.uintView[16] = this.current.color_theme;
    // 17: min_speed (f32)
    this.floatView[17] = this.current.min_speed;
    // 18: mouse_active (f32)
    this.floatView[18] = this.current.mouse_active;
    // 19: padding (f32)
    this.floatView[19] = 0.0;

    return this.bufferArray;
  }

  public createUniformNodes(): BoidSimUniformNodes {
    return {
      deltaTime: uniform(0.016),
      numParticles: uniform(this.current.num_particles),
      cohesionWeight: uniform(this.current.cohesion_weight),
      separationWeight: uniform(this.current.separation_weight),
      alignmentWeight: uniform(this.current.alignment_weight),
      maxSpeed: uniform(this.current.max_speed),
      minSpeed: uniform(this.current.min_speed),
      targetAttractionWeight: uniform(this.current.target_attraction_weight),
      aspectRatio: uniform(this.current.aspect_ratio),
      targetPos: uniform(new THREE.Vector2(this.current.target_pos[0], this.current.target_pos[1])),
      centroid: uniform(new THREE.Vector2(this.current.centroid[0], this.current.centroid[1])),
      vortexWeight: uniform(this.current.vortex_weight),
      repelRadius: uniform(this.current.repel_radius),
      alignRadius: uniform(this.current.align_radius),
      cohesionRadius: uniform(this.current.cohesion_radius),
      mouseActive: uniform(this.current.mouse_active),
    };
  }

  public syncToNodes(nodes: BoidSimUniformNodes): void {
    nodes.deltaTime.value = this.current.delta_time;
    nodes.numParticles.value = this.current.num_particles;
    nodes.cohesionWeight.value = this.current.cohesion_weight;
    nodes.separationWeight.value = this.current.separation_weight;
    nodes.alignmentWeight.value = this.current.alignment_weight;
    nodes.maxSpeed.value = this.current.max_speed;
    nodes.minSpeed.value = this.current.min_speed;
    nodes.targetAttractionWeight.value = this.current.target_attraction_weight;
    nodes.aspectRatio.value = this.current.aspect_ratio;
    nodes.targetPos.value.set(this.current.target_pos[0], this.current.target_pos[1]);
    nodes.centroid.value.set(this.current.centroid[0], this.current.centroid[1]);
    nodes.vortexWeight.value = this.current.vortex_weight;
    nodes.repelRadius.value = this.current.repel_radius;
    nodes.alignRadius.value = this.current.align_radius;
    nodes.cohesionRadius.value = this.current.cohesion_radius;
    nodes.mouseActive.value = this.current.mouse_active;
  }
}
