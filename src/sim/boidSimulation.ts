import * as THREE from 'three/webgpu';
import type { StorageBufferNode, ComputeNode } from 'three/webgpu';
import { storage } from 'three/tsl';
import { UniformManager } from './uniformManager';
import type { TelemetrySnapshot } from './types';
import {
  createBoidsComputeNode,
  type BoidSimUniformNodes,
} from '../shaders/boidsComputeTSL';
import {
  createBoidGeometry,
  createBoidsNodeMaterial,
} from '../shaders/boidsMaterialTSL';

export interface BoidSimOptions {
  canvas: HTMLCanvasElement;
  numParticles?: number;
  onTelemetry?: (telemetry: TelemetrySnapshot) => void;
  onError?: (errorMsg: string) => void;
}

export class BoidSimulation {
  public canvas: HTMLCanvasElement;
  public uniformManager: UniformManager;
  public numParticles: number;
  public isRunning: boolean = false;

  private renderer!: THREE.WebGPURenderer;
  private scene!: THREE.Scene;
  private camera!: THREE.OrthographicCamera;

  private maxCapacity: number = 131072;
  private particleBufferA!: THREE.StorageBufferAttribute;
  private particleBufferB!: THREE.StorageBufferAttribute;
  private storageNodeA!: StorageBufferNode<"vec4">;
  private storageNodeB!: StorageBufferNode<"vec4">;

  private uniformNodes!: BoidSimUniformNodes;
  private computeNodeAtoB!: ComputeNode;
  private computeNodeBtoA!: ComputeNode;
  private pingPongIndex: number = 0;

  private geometry!: THREE.BufferGeometry;
  private materialA!: THREE.MeshBasicNodeMaterial;
  private materialB!: THREE.MeshBasicNodeMaterial;
  private instancedMesh!: THREE.InstancedMesh;

  // Async Telemetry readback
  private sampleCount: number = 256;
  private isReadingBack: boolean = false;
  private lastReadbackTime: number = 0;
  private onTelemetry?: (telemetry: TelemetrySnapshot) => void;
  private onError?: (errorMsg: string) => void;

  // Frame timing & mouse tracking
  private lastFrameTime: number = 0;
  private frameCount: number = 0;
  private fpsUpdateTime: number = 0;
  private currentFps: number = 60;
  private animFrameId: number | null = null;

  private lastMousePos: [number, number] = [0, 0];
  private mousePos: [number, number] = [0, 0];
  private lastMouseMoveTime: number = 0;
  private isMouseDown: boolean = false;

  constructor(options: BoidSimOptions) {
    this.canvas = options.canvas;
    this.numParticles = options.numParticles ?? 65536;
    this.onTelemetry = options.onTelemetry;
    this.onError = options.onError;

    const aspect = Math.max(
      0.1,
      (this.canvas.clientWidth || window.innerWidth) / (this.canvas.clientHeight || window.innerHeight)
    );
    this.uniformManager = new UniformManager(this.numParticles, aspect);

    this.setupMouseEvents();
    this.setupResizeObserver();
  }

  public async init(): Promise<boolean> {
    if (!navigator.gpu) {
      const msg =
        'WebGPU is not supported in this browser. Please use Chrome 113+, Edge 113+, Safari 18+, or Firefox Nightly with WebGPU enabled.';
      this.onError?.(msg);
      return false;
    }

    try {
      // 1. Initialize Three.js WebGPU Renderer
      this.renderer = new THREE.WebGPURenderer({
        canvas: this.canvas,
        antialias: false, // MSAA disabled — 4x MSAA at 120Hz/2xDPR consumed ~33% of frame budget
        alpha: false,     // alphaMode: 'opaque' — required for macOS Metal display link to run at 120Hz
        powerPreference: 'high-performance',
      });
      await this.renderer.init();

      this.renderer.setClearColor(new THREE.Color(0.985, 0.983, 0.978), 1.0);
      this.resizeCanvas();

      // 2. Scene and Orthographic Camera
      this.scene = new THREE.Scene();
      this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 10);
      this.camera.position.z = 1;

      // 3. Seed initial particle buffers (max 131,072)
      const initialData = new Float32Array(this.maxCapacity * 4);
      const aspect = Math.max(0.1, this.uniformManager.current.aspect_ratio);

      for (let i = 0; i < this.maxCapacity; i++) {
        const idx = i * 4;
        const angle = Math.random() * Math.PI * 2;
        const radius = 0.05 + Math.sqrt(Math.random()) * 0.55;
        initialData[idx + 0] = Math.cos(angle) * radius * aspect; // pos.x
        initialData[idx + 1] = Math.sin(angle) * radius;          // pos.y

        const speed = 0.25 + Math.random() * 0.25;
        initialData[idx + 2] = -Math.sin(angle) * speed;         // vel.x
        initialData[idx + 3] = Math.cos(angle) * speed;          // vel.y
      }

      this.particleBufferA = new THREE.StorageBufferAttribute(initialData, 4);
      this.particleBufferB = new THREE.StorageBufferAttribute(new Float32Array(initialData), 4);

      this.storageNodeA = storage(this.particleBufferA, 'vec4', this.maxCapacity);
      this.storageNodeB = storage(this.particleBufferB, 'vec4', this.maxCapacity);

      // 4. Create TSL Uniform Nodes
      this.uniformNodes = this.uniformManager.createUniformNodes();

      // 5. Create TSL Compute Nodes for Ping-Pong Simulation
      this.computeNodeAtoB = createBoidsComputeNode(
        this.storageNodeA,
        this.storageNodeB,
        this.uniformNodes,
        this.numParticles
      );
      this.computeNodeBtoA = createBoidsComputeNode(
        this.storageNodeB,
        this.storageNodeA,
        this.uniformNodes,
        this.numParticles
      );

      // 6. Create Instanced Mesh with TSL Materials
      this.geometry = createBoidGeometry();
      this.materialA = createBoidsNodeMaterial(this.storageNodeA, this.uniformNodes);
      this.materialB = createBoidsNodeMaterial(this.storageNodeB, this.uniformNodes);

      this.instancedMesh = new THREE.InstancedMesh(this.geometry, this.materialA, this.maxCapacity);
      this.instancedMesh.count = this.numParticles;
      this.instancedMesh.frustumCulled = false;
      this.scene.add(this.instancedMesh);

      return true;
    } catch (err: any) {
      console.error('Failed to initialize WebGPU TSL simulation:', err);
      this.onError?.(`WebGPU TSL initialization error: ${err?.message || err}`);
      return false;
    }
  }

  public setParticleCount(count: number): void {
    const clamped = Math.max(1024, Math.min(this.maxCapacity, Math.floor(count / 64) * 64));
    this.numParticles = clamped;
    this.uniformManager.setParticleCount(clamped);

    if (this.instancedMesh) {
      this.instancedMesh.count = clamped;
    }
    if (this.computeNodeAtoB) {
      this.computeNodeAtoB.count = clamped;
    }
    if (this.computeNodeBtoA) {
      this.computeNodeBtoA.count = clamped;
    }
    if (this.uniformNodes) {
      this.uniformNodes.numParticles.value = clamped;
    }
  }

  public start(): void {
    if (this.isRunning) return;
    this.isRunning = true;
    this.lastFrameTime = performance.now();
    this.fpsUpdateTime = performance.now();
    this.frameCount = 0;
    this.animFrameId = requestAnimationFrame(this.loop);
  }

  public stop(): void {
    this.isRunning = false;
    if (this.animFrameId !== null) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
  }

  private loop = (): void => {
    if (!this.isRunning) return;

    const now = performance.now();
    const dt = Math.min((now - this.lastFrameTime) / 1000, 0.033);
    this.lastFrameTime = now;

    // Framerate tracking
    this.frameCount++;
    if (now - this.fpsUpdateTime >= 500) {
      this.currentFps = Math.round((this.frameCount * 1000) / (now - this.fpsUpdateTime));
      this.frameCount = 0;
      this.fpsUpdateTime = now;
    }

    this.frame(dt);
    this.animFrameId = requestAnimationFrame(this.loop);
  };

  private frame(dt: number): void {
    if (!this.renderer) return;

    // Check if mouse has stopped moving for > 1.5s
    const now = performance.now();
    if (now - this.lastMouseMoveTime > 1500 && !this.isMouseDown) {
      this.uniformManager.updateTargetPos(this.mousePos[0], this.mousePos[1], false);
    }

    // 1. Update uniforms CPU-side and sync to TSL Uniform Nodes
    this.uniformManager.update(dt);
    this.uniformManager.syncToNodes(this.uniformNodes);

    // 2. Dispatch TSL Compute Pass & Ping-Pong
    let activeBuffer: THREE.StorageBufferAttribute;
    if (this.pingPongIndex === 0) {
      this.renderer.compute(this.computeNodeAtoB);
      this.instancedMesh.material = this.materialB;
      activeBuffer = this.particleBufferB;
      this.pingPongIndex = 1;
    } else {
      this.renderer.compute(this.computeNodeBtoA);
      this.instancedMesh.material = this.materialA;
      activeBuffer = this.particleBufferA;
      this.pingPongIndex = 0;
    }

    // 3. Render Pass
    this.renderer.render(this.scene, this.camera);

    // 4. Asynchronous Telemetry Readback (~3 Hz)
    if (!this.isReadingBack && now - this.lastReadbackTime > 300) {
      this.lastReadbackTime = now;
      this.isReadingBack = true;
      this.readTelemetry(activeBuffer);
    }
  }

  private async readTelemetry(buffer: THREE.StorageBufferAttribute): Promise<void> {
    try {
      const sampleByteSize = this.sampleCount * 16; // 256 particles * 16 bytes = 4096 bytes
      const arrayBuffer = await this.renderer.getArrayBufferAsync(buffer, null, 0, sampleByteSize);
      const floatData = new Float32Array(arrayBuffer);

      this.processTelemetryData(floatData);
    } catch (err) {
      console.warn('Telemetry readback notice:', err);
    } finally {
      this.isReadingBack = false;
    }
  }

  private processTelemetryData(floatData: Float32Array): void {
    const count = Math.min(this.sampleCount, Math.floor(floatData.length / 4));
    if (count <= 0) return;

    let sumX = 0;
    let sumY = 0;
    let sumSpeed = 0;

    for (let i = 0; i < count; i++) {
      const idx = i * 4;
      const px = floatData[idx + 0];
      const py = floatData[idx + 1];
      const vx = floatData[idx + 2];
      const vy = floatData[idx + 3];

      sumX += px;
      sumY += py;
      sumSpeed += Math.sqrt(vx * vx + vy * vy);
    }

    const meanX = sumX / count;
    const meanY = sumY / count;
    const avgSpeed = sumSpeed / count;

    // Spatial dispersion (standard deviation from centroid)
    let varianceSum = 0;
    for (let i = 0; i < count; i++) {
      const idx = i * 4;
      const dx = floatData[idx + 0] - meanX;
      const dy = floatData[idx + 1] - meanY;
      varianceSum += dx * dx + dy * dy;
    }
    const dispersion = Math.sqrt(varianceSum / count);

    // Inform uniform manager of live centroid
    this.uniformManager.updateCentroid([meanX, meanY]);

    // Distance to cursor
    const distToCursor = Math.hypot(this.mousePos[0] - meanX, this.mousePos[1] - meanY);

    // Cursor speed estimation
    const mouseDelta = Math.hypot(this.mousePos[0] - this.lastMousePos[0], this.mousePos[1] - this.lastMousePos[1]);
    this.lastMousePos = [this.mousePos[0], this.mousePos[1]];

    let cursorSpeed: TelemetrySnapshot['cursor_speed'] = 'static';
    if (mouseDelta > 0.28) {
      cursorSpeed = 'erratic';
    } else if (mouseDelta > 0.05) {
      cursorSpeed = 'fast_approach';
    } else if (mouseDelta > 0.005) {
      cursorSpeed = 'slow_drift';
    }

    // Viewport boundary pressure
    const aspect = this.uniformManager.current.aspect_ratio;
    const boundXDist = aspect - Math.abs(meanX);
    const boundYDist = 1.0 - Math.abs(meanY);
    const minBoundDist = Math.min(boundXDist, boundYDist);

    let boundaryPressure: TelemetrySnapshot['viewport_boundary_pressure'] = 'low';
    if (minBoundDist < 0.2) {
      boundaryPressure = 'high';
    } else if (minBoundDist < 0.45) {
      boundaryPressure = 'moderate';
    }

    const snapshot: TelemetrySnapshot = {
      centroid: [Number(meanX.toFixed(3)), Number(meanY.toFixed(3))],
      avg_speed: Number(avgSpeed.toFixed(3)),
      swarm_dispersion: Number(dispersion.toFixed(3)),
      distance_to_cursor: Number(distToCursor.toFixed(3)),
      cursor_speed: cursorSpeed,
      viewport_boundary_pressure: boundaryPressure,
      fps: this.currentFps,
      num_particles: this.numParticles,
    };

    this.onTelemetry?.(snapshot);
  }

  private setupMouseEvents(): void {
    const updateMouse = (clientX: number, clientY: number, active: boolean) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = ((clientX - rect.left) / rect.width) * 2 - 1;
      const y = -(((clientY - rect.top) / rect.height) * 2 - 1);
      const aspect = this.uniformManager.current.aspect_ratio;

      this.mousePos = [x * aspect, y];
      this.lastMouseMoveTime = performance.now();
      this.uniformManager.updateTargetPos(this.mousePos[0], this.mousePos[1], active);
    };

    window.addEventListener('mousemove', (e) => {
      updateMouse(e.clientX, e.clientY, true);
    });

    this.canvas.addEventListener('mousedown', () => {
      this.isMouseDown = true;
    });

    window.addEventListener('mouseup', () => {
      this.isMouseDown = false;
    });

    this.canvas.addEventListener(
      'touchmove',
      (e) => {
        if (e.touches.length > 0) {
          updateMouse(e.touches[0].clientX, e.touches[0].clientY, true);
        }
      },
      { passive: true }
    );
  }

  private setupResizeObserver(): void {
    window.addEventListener('resize', () => {
      this.resizeCanvas();
    });
  }

  private resizeCanvas(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor((this.canvas.clientWidth || window.innerWidth) * dpr));
    const height = Math.max(1, Math.floor((this.canvas.clientHeight || window.innerHeight) * dpr));

    if (this.canvas.width !== width || this.canvas.height !== height) {
      const aspect = width / height;
      this.uniformManager.setAspectRatio(aspect);
      if (this.uniformNodes) {
        this.uniformNodes.aspectRatio.value = aspect;
      }
      if (this.renderer) {
        this.renderer.setSize(width / dpr, height / dpr, false);
        this.renderer.setPixelRatio(dpr);
      }
    }
  }
}
