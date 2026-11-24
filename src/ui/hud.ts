import type { InstinctType, JevDecision, TelemetrySnapshot } from '../sim/types';
import { INSTINCT_PROFILES } from '../sim/uniformManager';
import type { JevDirector } from '../services/jevDirector';
import type { BoidSimulation } from '../sim/boidSimulation';

export class HudController {
  private container: HTMLElement;
  private sim: BoidSimulation;
  private director: JevDirector;

  // Cached DOM elements
  private elFps!: HTMLElement;
  private elParticles!: HTMLElement;
  private elCentroid!: HTMLElement;
  private elSpeed!: HTMLElement;
  private elDispersion!: HTMLElement;
  private elCursorDist!: HTMLElement;
  private elCursorSpeed!: HTMLElement;
  private elBoundaryPressure!: HTMLElement;

  private elInstinctBadge!: HTMLElement;
  private elInstinctDesc!: HTMLElement;
  private elConfidenceVal!: HTMLElement;
  private elConfidenceBar!: HTMLElement;
  private elAgitationValue!: HTMLElement;
  private elAgitationBar!: HTMLElement;
  private elBtnPolarityAttract!: HTMLButtonElement;
  private elBtnPolarityRepel!: HTMLButtonElement;
  private elSourceBadge!: HTMLElement;
  private elLatencyVal!: HTMLElement;

  private elToggleModeBtn!: HTMLButtonElement;
  private elApiKeyModal!: HTMLElement;
  private elApiKeyInput!: HTMLInputElement;

  // Collapsible state (start minimized by default)
  private isCognitiveCollapsed = true;
  private isTelemetryCollapsed = true;
  private isZenMode = false;

  constructor(container: HTMLElement, sim: BoidSimulation, director: JevDirector) {
    this.container = container;
    this.sim = sim;
    this.director = director;

    this.renderHtml();
    this.bindEvents();
    this.updateModeDisplay();
  }

  private renderHtml(): void {
    this.container.innerHTML = `
      <!-- Top Bar / Header -->
      <header class="hud-header" id="hud-header">
        <div class="hud-brand">
          <div class="brand-glyph"></div>
          <div>
            <div class="brand-title">JEV HIVEMIND // BOIDS</div>
            <div class="brand-subtitle">WEBGPU COMPUTE SHADER + TYPESAFE SYSTEM ONE</div>
          </div>
        </div>

        <div class="hud-top-actions">
          <button class="zen-toggle-btn" id="btn-zen-mode" title="Toggle Zen Mode (Press 'H')">
            <span class="zen-icon">👁</span> ZEN MODE
          </button>
          <button class="api-key-btn" id="btn-api-key" title="Configure TypeSafe API Key">
            API KEY
          </button>
          <div class="director-mode-toggle">
            <span class="mode-label">ENGINE:</span>
            <button class="mode-switch-btn" id="btn-toggle-mode">
              <span class="indicator-dot" id="mode-dot"></span>
              <span id="mode-text">MOCK AI</span>
            </button>
          </div>
        </div>
      </header>

      <!-- Zen Mode Floating Restore Button -->
      <button class="zen-restore-btn" id="btn-restore-hud" title="Exit Zen Mode (Press 'H')">
        <span>👁</span> RESTORE HUD
      </button>

      <!-- Bottom HUD Dual Panels -->
      <main class="hud-layout" id="hud-layout">
        <!-- Left: Cognitive Decision Director -->
        <section class="hud-panel cognitive-panel collapsed" id="panel-cognitive">
          <div class="panel-header">
            <div class="panel-title-row">
              <span class="panel-tag">COGNITIVE LOOP</span>
              <span class="source-tag mock" id="eval-source">MOCK AI</span>
            </div>
            <div class="panel-header-controls">
              <span class="latency-tag"><span id="eval-latency">1</span>ms</span>
              <button class="panel-collapse-btn" id="btn-collapse-cognitive" title="Toggle Panel">+</button>
            </div>
          </div>

          <div class="panel-collapsible-body" id="cognitive-body">
            <!-- Active Instinct Badge -->
            <div class="instinct-card">
              <div class="instinct-header">
                <span class="instinct-label">ACTIVE INSTINCT</span>
                <span class="instinct-badge murmuration" id="instinct-badge">MURMURATION</span>
              </div>
              <p class="instinct-desc" id="instinct-desc">
                Calm, expansive sweeping arcs; high cohesion and smooth alignment.
              </p>
            </div>

            <!-- Confidence Meter -->
            <div class="meter-group">
              <div class="meter-label-row">
                <span class="meter-label">AI CONFIDENCE</span>
                <span class="meter-val" id="confidence-val">90%</span>
              </div>
              <div class="meter-track">
                <div class="meter-fill confidence-fill" id="confidence-bar" style="width: 90%;"></div>
              </div>
            </div>

            <!-- Agitation Stress Bar -->
            <div class="meter-group">
              <div class="meter-label-row">
                <span class="meter-label">SWARM AGITATION (STRESS)</span>
                <span class="meter-val" id="agitation-val">LEVEL 1 / 5</span>
              </div>
              <div class="agitation-segments" id="agitation-bar">
                <div class="seg active"></div>
                <div class="seg"></div>
                <div class="seg"></div>
                <div class="seg"></div>
                <div class="seg"></div>
              </div>
            </div>

            <!-- Polarity / Cursor Force Controls -->
            <div class="status-row polarity-row">
              <span class="status-label">CURSOR FORCE:</span>
              <div class="polarity-toggle-group">
                <button class="polarity-btn active" id="btn-polarity-attract" data-polarity="attractor" title="Cursor pulls boids toward it">
                  <span class="polarity-icon">🧲</span> ATTRACTOR
                </button>
                <button class="polarity-btn" id="btn-polarity-repel" data-polarity="repulsor" title="Cursor pushes boids away in smooth waves">
                  <span class="polarity-icon">💥</span> REPULSOR
                </button>
              </div>
            </div>

            <!-- Manual Trigger Presets -->
            <div class="preset-triggers">
              <div class="section-label">TRIGGER INSTINCT OVERRIDE</div>
              <div class="preset-btn-grid">
                <button class="preset-btn" data-instinct="murmuration">MURMURATION</button>
                <button class="preset-btn" data-instinct="scatter_panic">PANIC SCATTER</button>
                <button class="preset-btn" data-instinct="vortex_coil">VORTEX COIL</button>
                <button class="preset-btn" data-instinct="predator_mob">PREDATOR MOB</button>
              </div>
            </div>
          </div>
        </section>

        <!-- Right: Real-time GPU Telemetry & Particle Controls -->
        <section class="hud-panel telemetry-panel collapsed" id="panel-telemetry">
          <div class="panel-header">
            <div class="panel-title-row">
              <span class="panel-tag">GPU TELEMETRY</span>
            </div>
            <div class="panel-header-controls">
              <span class="perf-tag"><span id="fps-val">60</span> FPS</span>
              <button class="panel-collapse-btn" id="btn-collapse-telemetry" title="Toggle Panel">+</button>
            </div>
          </div>

          <div class="panel-collapsible-body" id="telemetry-body">
            <div class="telemetry-grid">
              <div class="stat-box">
                <div class="stat-label">PARTICLES</div>
                <div class="stat-val" id="particle-count-val">65,536</div>
              </div>
              <div class="stat-box">
                <div class="stat-label">AVG SPEED</div>
                <div class="stat-val" id="avg-speed-val">0.420</div>
              </div>
              <div class="stat-box">
                <div class="stat-label">DISPERSION</div>
                <div class="stat-val" id="dispersion-val">0.280</div>
              </div>
              <div class="stat-box">
                <div class="stat-label">CURSOR DIST</div>
                <div class="stat-val" id="cursor-dist-val">0.340</div>
              </div>
              <div class="stat-box">
                <div class="stat-label">CURSOR SPEED</div>
                <div class="stat-val highlight" id="cursor-speed-val">STATIC</div>
              </div>
              <div class="stat-box">
                <div class="stat-label">CENTROID</div>
                <div class="stat-val" id="centroid-val">[0.00, 0.00]</div>
              </div>
              <div class="stat-box full-width">
                <div class="stat-label">BOUNDARY PRESSURE</div>
                <div class="stat-val" id="boundary-pressure-val">LOW</div>
              </div>
            </div>

            <!-- Particle Count Selector -->
            <div class="particle-selector-group">
              <div class="section-label">SWARM DENSITY</div>
              <div class="density-buttons">
                <button class="density-btn" data-count="32768">32K</button>
                <button class="density-btn active" data-count="65536">65K</button>
                <button class="density-btn" data-count="98304">98K</button>
                <button class="density-btn" data-count="131072">131K</button>
              </div>
            </div>

            <!-- Interaction Guidance Note -->
            <div class="guidance-note">
              <span>Move cursor to guide • Toggle Attractor/Repulsor button to switch polarity</span>
            </div>
          </div>
        </section>
      </main>

      <!-- API Key Modal -->
      <div class="modal-backdrop" id="api-modal">
        <div class="modal-dialog">
          <div class="modal-header">
            <h3 class="modal-title">TypeSafe System One API Key</h3>
            <button class="modal-close-btn" id="btn-close-modal">✕</button>
          </div>
          <p class="modal-description">
            Enter your <code>TYPESAFE_API_KEY</code> to enable live AI cognitive directing.
            When active, the swarm consults TypeSafe System One every 600ms to determine
            collective flight instinct, agitation scores, and threat response.
          </p>
          <input
            type="password"
            class="modal-input"
            id="input-api-key"
            placeholder="ts_live_..."
            autocomplete="off"
          />
          <div class="modal-footer">
            <button class="hud-btn hud-btn-subtle" id="btn-clear-key">Clear</button>
            <button class="hud-btn hud-btn-primary" id="btn-save-key">Save & Activate</button>
          </div>
        </div>
      </div>
    `;

    // Cache elements
    this.elFps = document.getElementById('fps-val')!;
    this.elParticles = document.getElementById('particle-count-val')!;
    this.elCentroid = document.getElementById('centroid-val')!;
    this.elSpeed = document.getElementById('avg-speed-val')!;
    this.elDispersion = document.getElementById('dispersion-val')!;
    this.elCursorDist = document.getElementById('cursor-dist-val')!;
    this.elCursorSpeed = document.getElementById('cursor-speed-val')!;
    this.elBoundaryPressure = document.getElementById('boundary-pressure-val')!;

    this.elInstinctBadge = document.getElementById('instinct-badge')!;
    this.elInstinctDesc = document.getElementById('instinct-desc')!;
    this.elConfidenceVal = document.getElementById('confidence-val')!;
    this.elConfidenceBar = document.getElementById('confidence-bar')!;
    this.elAgitationValue = document.getElementById('agitation-val')!;
    this.elAgitationBar = document.getElementById('agitation-bar')!;
    this.elBtnPolarityAttract = document.getElementById('btn-polarity-attract') as HTMLButtonElement;
    this.elBtnPolarityRepel = document.getElementById('btn-polarity-repel') as HTMLButtonElement;
    this.elSourceBadge = document.getElementById('eval-source')!;
    this.elLatencyVal = document.getElementById('eval-latency')!;

    this.elToggleModeBtn = document.getElementById('btn-toggle-mode') as HTMLButtonElement;
    this.elApiKeyModal = document.getElementById('api-modal')!;
    this.elApiKeyInput = document.getElementById('input-api-key') as HTMLInputElement;
  }

  private bindEvents(): void {
    // Toggle Mode Button
    this.elToggleModeBtn.addEventListener('click', () => {
      const isCurrentlyMock = this.director.getIsMockMode();
      if (isCurrentlyMock && !this.director.getApiKey()) {
        this.openApiKeyModal();
      } else {
        this.director.setMode(!isCurrentlyMock);
        this.updateModeDisplay();
      }
    });

    // API Key Modal
    const btnKey = document.getElementById('btn-api-key')!;
    btnKey.addEventListener('click', () => this.openApiKeyModal());

    const btnClose = document.getElementById('btn-close-modal')!;
    btnClose.addEventListener('click', () => this.closeApiKeyModal());

    const btnSave = document.getElementById('btn-save-key')!;
    btnSave.addEventListener('click', () => {
      const key = this.elApiKeyInput.value.trim();
      this.director.setApiKey(key);
      this.closeApiKeyModal();
      this.updateModeDisplay();
    });

    const btnClear = document.getElementById('btn-clear-key')!;
    btnClear.addEventListener('click', () => {
      this.elApiKeyInput.value = '';
      this.director.setApiKey('');
      this.closeApiKeyModal();
      this.updateModeDisplay();
    });

    // Cursor Force Polarity Buttons
    const setPolarity = (mode: 'attractor' | 'repulsor') => {
      this.sim.uniformManager.setPolarityMode(mode);
      this.updatePolarityUI(mode);
    };

    this.elBtnPolarityAttract?.addEventListener('click', () => setPolarity('attractor'));
    this.elBtnPolarityRepel?.addEventListener('click', () => setPolarity('repulsor'));

    // Preset instinct triggers
    const presetBtns = this.container.querySelectorAll<HTMLButtonElement>('.preset-btn');
    presetBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const instinct = btn.getAttribute('data-instinct') as InstinctType;
        if (instinct) {
          this.director.triggerManualInstinct(instinct);
        }
      });
    });

    // Density buttons
    const densityBtns = this.container.querySelectorAll<HTMLButtonElement>('.density-btn');
    densityBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        densityBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const count = parseInt(btn.getAttribute('data-count') || '65536', 10);
        this.sim.setParticleCount(count);
      });
    });

    // Collapsible Cognitive Panel
    const btnCollapseCog = document.getElementById('btn-collapse-cognitive')!;
    const panelCog = document.getElementById('panel-cognitive')!;
    btnCollapseCog.addEventListener('click', () => {
      this.isCognitiveCollapsed = !this.isCognitiveCollapsed;
      panelCog.classList.toggle('collapsed', this.isCognitiveCollapsed);
      btnCollapseCog.textContent = this.isCognitiveCollapsed ? '+' : '−';
    });

    // Collapsible Telemetry Panel
    const btnCollapseTel = document.getElementById('btn-collapse-telemetry')!;
    const panelTel = document.getElementById('panel-telemetry')!;
    btnCollapseTel.addEventListener('click', () => {
      this.isTelemetryCollapsed = !this.isTelemetryCollapsed;
      panelTel.classList.toggle('collapsed', this.isTelemetryCollapsed);
      btnCollapseTel.textContent = this.isTelemetryCollapsed ? '+' : '−';
    });

    // Zen Mode (Hide all HUD elements for clean view)
    const btnZen = document.getElementById('btn-zen-mode')!;
    const btnRestore = document.getElementById('btn-restore-hud')!;

    const toggleZenMode = () => {
      this.isZenMode = !this.isZenMode;
      this.container.classList.toggle('zen-mode', this.isZenMode);
    };

    btnZen.addEventListener('click', toggleZenMode);
    btnRestore.addEventListener('click', toggleZenMode);

    // Keyboard shortcut: 'H' toggles Zen Mode
    window.addEventListener('keydown', (e) => {
      // Don't trigger if typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }
      if (e.key === 'h' || e.key === 'H') {
        toggleZenMode();
      }
    });
  }

  public updatePolarityUI(mode: 'attractor' | 'repulsor'): void {
    if (!this.elBtnPolarityAttract || !this.elBtnPolarityRepel) return;
    if (mode === 'repulsor') {
      this.elBtnPolarityRepel.classList.add('active');
      this.elBtnPolarityAttract.classList.remove('active');
    } else {
      this.elBtnPolarityAttract.classList.add('active');
      this.elBtnPolarityRepel.classList.remove('active');
    }
  }

  private openApiKeyModal(): void {
    this.elApiKeyInput.value = this.director.getApiKey();
    this.elApiKeyModal.classList.add('active');
  }

  private closeApiKeyModal(): void {
    this.elApiKeyModal.classList.remove('active');
  }

  private updateModeDisplay(): void {
    const isMock = this.director.getIsMockMode();
    const hasKey = Boolean(this.director.getApiKey());

    const dot = document.getElementById('mode-dot')!;
    const text = document.getElementById('mode-text')!;
    const btnKey = document.getElementById('btn-api-key')!;

    if (isMock) {
      dot.className = 'indicator-dot mock';
      text.textContent = 'MOCK AI';
      this.elToggleModeBtn.title = hasKey
        ? 'Click to switch to Live TypeSafe System One'
        : 'Click to configure API Key';
      btnKey.textContent = hasKey ? 'KEY (MOCK)' : 'SET KEY';
      btnKey.classList.toggle('active', hasKey);
    } else {
      dot.className = 'indicator-dot live';
      text.textContent = 'LIVE JEV';
      this.elToggleModeBtn.title = 'Click to switch to Mock AI';
      btnKey.textContent = 'KEY (ACTIVE)';
      btnKey.classList.add('active');
    }
  }

  public updateTelemetry(t: TelemetrySnapshot): void {
    this.elFps.textContent = `${t.fps}`;
    this.elParticles.textContent = t.num_particles.toLocaleString();
    this.elSpeed.textContent = t.avg_speed.toFixed(3);
    this.elDispersion.textContent = t.swarm_dispersion.toFixed(3);
    this.elCursorDist.textContent = t.distance_to_cursor.toFixed(3);
    this.elCursorSpeed.textContent = t.cursor_speed.toUpperCase();
    this.elCentroid.textContent = `[${t.centroid[0].toFixed(2)}, ${t.centroid[1].toFixed(2)}]`;

    this.elBoundaryPressure.textContent = t.viewport_boundary_pressure.toUpperCase();
    this.elBoundaryPressure.className = `stat-val ${
      t.viewport_boundary_pressure === 'high' ? 'warn' : ''
    }`;
  }

  public updateDecision(d: JevDecision): void {
    const profile = INSTINCT_PROFILES[d.instinct.value];
    if (profile) {
      this.elInstinctBadge.textContent = profile.label.toUpperCase();
      this.elInstinctBadge.className = `instinct-badge ${d.instinct.value}`;
      this.elInstinctDesc.textContent = profile.description;
    }

    const confPct = Math.round(d.instinct.confidence * 100);
    this.elConfidenceVal.textContent = `${confPct}%`;
    this.elConfidenceBar.style.width = `${confPct}%`;

    this.elAgitationValue.textContent = `LEVEL ${d.agitation.value} / 5`;
    const segs = this.elAgitationBar.querySelectorAll('.seg');
    segs.forEach((seg, idx) => {
      if (idx < d.agitation.value) {
        seg.className = `seg active level-${d.agitation.value}`;
      } else {
        seg.className = 'seg';
      }
    });

    // Update polarity button active indicator
    this.updatePolarityUI(this.sim.uniformManager.polarityMode);

    this.elSourceBadge.className = `source-tag ${d.source}`;
    this.elSourceBadge.textContent = d.source === 'live' ? 'LIVE JEV' : 'MOCK AI';
    this.elLatencyVal.textContent = `${d.latencyMs}ms`;
  }
}
