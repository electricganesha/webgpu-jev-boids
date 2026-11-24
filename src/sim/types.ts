export type InstinctType = 'murmuration' | 'scatter_panic' | 'vortex_coil' | 'predator_mob';

export interface InstinctProfile {
  label: string;
  description: string;
  cohesion_weight: number;
  separation_weight: number;
  alignment_weight: number;
  max_speed: number;
  min_speed: number;
  target_attraction_weight: number;
  vortex_weight: number;
  repel_radius: number;
  align_radius: number;
  cohesion_radius: number;
  color_theme: number;
}

export interface SimUniforms {
  delta_time: number;
  num_particles: number;
  cohesion_weight: number;
  separation_weight: number;
  alignment_weight: number;
  max_speed: number;
  target_attraction_weight: number;
  aspect_ratio: number;
  target_pos: [number, number];
  centroid: [number, number];
  vortex_weight: number;
  repel_radius: number;
  align_radius: number;
  cohesion_radius: number;
  color_theme: number;
  min_speed: number;
  mouse_active: number;
}

export interface TelemetrySnapshot {
  centroid: [number, number];
  avg_speed: number;
  swarm_dispersion: number;
  distance_to_cursor: number;
  cursor_speed: 'static' | 'slow_drift' | 'fast_approach' | 'erratic';
  viewport_boundary_pressure: 'low' | 'moderate' | 'high';
  fps: number;
  num_particles: number;
}

export interface JevDecision {
  instinct: {
    value: InstinctType;
    confidence: number;
  };
  agitation: {
    value: number; // 1 to 5
    confidence: number;
  };
  invert_polarity: {
    value: boolean;
    probability: number;
  };
  source: 'live' | 'mock';
  latencyMs: number;
  timestamp: number;
}
