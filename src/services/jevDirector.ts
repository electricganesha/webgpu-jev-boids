import { TypeSafeClient, choice, score, noul } from '@typesafe-ai/sdk';
import type { InstinctType, JevDecision, TelemetrySnapshot } from '../sim/types';
import type { UniformManager } from '../sim/uniformManager';

export interface JevDirectorOptions {
  uniformManager: UniformManager;
  onDecision?: (decision: JevDecision) => void;
  intervalMs?: number;
}

export class JevDirector {
  private uniformManager: UniformManager;
  private onDecision?: (decision: JevDecision) => void;
  private intervalMs: number;

  private latestTelemetry: TelemetrySnapshot | null = null;
  private timerId: number | null = null;
  private isEvaluating: boolean = false;
  private isMockMode: boolean = true;
  private apiKey: string = '';

  private client: TypeSafeClient | null = null;
  private manualOverrideUntil: number = 0;

  constructor(options: JevDirectorOptions) {
    this.uniformManager = options.uniformManager;
    this.onDecision = options.onDecision;
    this.intervalMs = options.intervalMs ?? 600;

    // Read API key from env or localStorage
    const envKey = ((import.meta as any).env?.VITE_TYPESAFE_API_KEY || '').trim();
    const storedKey = typeof localStorage !== 'undefined' ? (localStorage.getItem('typesafe_api_key') || '').trim() : '';
    this.apiKey = storedKey || envKey;

    if (this.apiKey.length > 0) {
      this.initClient(this.apiKey);
      this.isMockMode = false;
    } else {
      this.isMockMode = true;
    }
  }

  public setApiKey(key: string): void {
    this.apiKey = key.trim();
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('typesafe_api_key', this.apiKey);
    }

    if (this.apiKey.length > 0) {
      this.initClient(this.apiKey);
      this.isMockMode = false;
    } else {
      this.client = null;
      this.isMockMode = true;
    }
  }

  public getApiKey(): string {
    return this.apiKey;
  }

  public setMode(mock: boolean): void {
    this.isMockMode = mock;
  }

  public getIsMockMode(): boolean {
    return this.isMockMode;
  }

  private initClient(key: string): void {
    try {
      const isDev = (import.meta as any).env?.DEV;
      this.client = new TypeSafeClient({
        apiKey: key,
        baseURL: isDev ? '/typesafe-proxy' : undefined,
        dangerouslyAllowBrowser: true,
        timeout: 4000,
      });
    } catch (e) {
      console.warn('Could not initialize TypeSafeClient:', e);
      this.client = null;
      this.isMockMode = true;
    }
  }

  public start(): void {
    if (this.timerId !== null) return;
    this.timerId = window.setInterval(() => {
      this.tick();
    }, this.intervalMs);
  }

  public stop(): void {
    if (this.timerId !== null) {
      clearInterval(this.timerId);
      this.timerId = null;
    }
  }

  public ingestTelemetry(telemetry: TelemetrySnapshot): void {
    this.latestTelemetry = telemetry;
  }

  private async tick(): Promise<void> {
    // Drop execution if an evaluation is already in flight, telemetry is missing, or manual override is active
    if (this.isEvaluating || !this.latestTelemetry || Date.now() < this.manualOverrideUntil) {
      return;
    }

    this.isEvaluating = true;
    const startTime = performance.now();

    try {
      if (!this.isMockMode && this.client) {
        await this.evaluateLive(this.latestTelemetry, startTime);
      } else {
        this.evaluateMock(this.latestTelemetry, startTime);
      }
    } catch (err) {
      console.warn('Live Jev evaluation failed, falling back to mock heuristic:', err);
      this.evaluateMock(this.latestTelemetry, startTime);
    } finally {
      this.isEvaluating = false;
    }
  }

  private async evaluateLive(telemetry: TelemetrySnapshot, startTime: number): Promise<void> {
    if (!this.client) throw new Error('Client not initialized');

    const statePayload = {
      swarm_dispersion: telemetry.swarm_dispersion,
      average_speed: telemetry.avg_speed,
      distance_to_cursor: telemetry.distance_to_cursor,
      cursor_speed: telemetry.cursor_speed,
      viewport_boundary_pressure: telemetry.viewport_boundary_pressure,
    };

    const response = await this.client.systemOne({
      state: statePayload,
      questions: {
        instinct: choice(
          'Select the optimal collective swarm instinct based on telemetry dynamics',
          {
            vortex_coil: 'Autonomous self-organizing orbital coil when idle, static, or approaching boundaries; graceful circular flight.',
            murmuration: 'Calm, expansive sweeping arcs when actively migrating across open canvas.',
            scatter_panic: 'Explosive dispersal; high separation, chaotic bursts, repelled by cursor.',
            predator_mob: 'Aggressive convergence directly toward the cursor target.',
          }
        ),
        agitation: score('Evaluates swarm stress level', [
          '1: Serene and tranquil',
          '2: Attentive and aware',
          '3: Heightened activity',
          '4: Alarmed agitation',
          '5: Critical survival panic',
        ]),
        invert_polarity: noul('Whether the cursor becomes a repulsor instead of an attractor'),
      },
    });

    const answers = response.answers as any;
    const instinctVal = (answers.instinct?.choice || 'murmuration') as InstinctType;
    const instinctConf = answers.instinct?.confidence ?? 0.85;

    // Parse score from answer
    const rawScore = answers.agitation?.score;
    const agitationScore = typeof rawScore === 'number' ? Math.round(rawScore) + 1 : 2;
    const agitationConf = answers.agitation?.confidence ?? 0.8;

    // Parse Noul probability
    const noulVal = answers.invert_polarity?.noul ?? 0.0;
    const invert = noulVal > 0.5;

    const latency = Math.round(performance.now() - startTime);

    const decision: JevDecision = {
      instinct: {
        value: instinctVal,
        confidence: Number(instinctConf.toFixed(3)),
      },
      agitation: {
        value: Math.min(5, Math.max(1, agitationScore)),
        confidence: Number(agitationConf.toFixed(3)),
      },
      invert_polarity: {
        value: invert,
        probability: Number(noulVal.toFixed(3)),
      },
      source: 'live',
      latencyMs: latency,
      timestamp: Date.now(),
    };

    this.applyDecision(decision);
  }

  private evaluateMock(telemetry: TelemetrySnapshot, startTime: number): void {
    const { distance_to_cursor, cursor_speed, swarm_dispersion, viewport_boundary_pressure } = telemetry;

    let chosenInstinct: InstinctType = 'murmuration';
    let agitationLevel = 1;
    let confidence = 0.85;
    let invertPolarity = false;
    let invertProb = 0.05;

    // Emergent heuristic decision logic based on physical telemetry
    if ((cursor_speed === 'erratic' && distance_to_cursor < 0.6) || (cursor_speed === 'fast_approach' && distance_to_cursor < 0.4)) {
      // Swarm senses incoming threat in vicinity -> evasive murmuration
      chosenInstinct = 'scatter_panic';
      agitationLevel = distance_to_cursor < 0.25 ? 4 : 3;
      confidence = 0.88;
      invertPolarity = true;
      invertProb = 0.88;
    } else if (distance_to_cursor < 0.22 && cursor_speed === 'static' && this.uniformManager.current.mouse_active > 0.5) {
      // Active cursor placed right inside swarm -> gentle attractor mob
      chosenInstinct = 'predator_mob';
      agitationLevel = 2;
      confidence = 0.84;
      invertPolarity = false;
      invertProb = 0.12;
    } else if (swarm_dispersion > 0.40 || viewport_boundary_pressure === 'high' || viewport_boundary_pressure === 'moderate') {
      // Swarm dispersing or approaching boundary walls -> vortex coil to bundle up and circle inward
      chosenInstinct = 'vortex_coil';
      agitationLevel = 2;
      confidence = 0.88;
      invertPolarity = false;
      invertProb = 0.15;
    } else if (cursor_speed === 'static' || cursor_speed === 'slow_drift') {
      // Swarm left by themselves or drifting peacefully -> autonomous self-organizing vortex coil!
      chosenInstinct = 'vortex_coil';
      agitationLevel = 1;
      confidence = 0.92;
      invertPolarity = false;
      invertProb = 0.08;
    } else {
      // Calm, expansive cruising across open space
      chosenInstinct = 'murmuration';
      agitationLevel = 1;
      confidence = 0.85;
      invertPolarity = false;
      invertProb = 0.05;
    }

    // Add slight natural variance to confidence
    const jitter = (Math.random() - 0.5) * 0.08;
    confidence = Math.min(0.99, Math.max(0.65, confidence + jitter));

    const latency = Math.max(1, Math.round(performance.now() - startTime));

    const decision: JevDecision = {
      instinct: {
        value: chosenInstinct,
        confidence: Number(confidence.toFixed(3)),
      },
      agitation: {
        value: agitationLevel,
        confidence: Number((0.8 + Math.random() * 0.15).toFixed(3)),
      },
      invert_polarity: {
        value: invertPolarity,
        probability: Number(invertProb.toFixed(3)),
      },
      source: 'mock',
      latencyMs: latency,
      timestamp: Date.now(),
    };

    this.applyDecision(decision);
  }

  private applyDecision(decision: JevDecision): void {
    // 1. Update UniformManager target weights according to selected instinct
    this.uniformManager.setInstinct(
      decision.instinct.value,
      decision.instinct.confidence,
      decision.invert_polarity.value
    );

    // 2. Notify subscribers (HUD telemetry)
    this.onDecision?.(decision);
  }

  public triggerManualInstinct(instinct: InstinctType, holdDurationMs: number = 5000): void {
    this.manualOverrideUntil = Date.now() + holdDurationMs;
    const decision: JevDecision = {
      instinct: {
        value: instinct,
        confidence: 0.99,
      },
      agitation: {
        value: instinct === 'scatter_panic' ? 5 : instinct === 'predator_mob' ? 4 : 2,
        confidence: 0.95,
      },
      invert_polarity: {
        value: instinct === 'scatter_panic',
        probability: instinct === 'scatter_panic' ? 0.99 : 0.05,
      },
      source: this.isMockMode ? 'mock' : 'live',
      latencyMs: 0,
      timestamp: Date.now(),
    };

    this.applyDecision(decision);
  }
}
