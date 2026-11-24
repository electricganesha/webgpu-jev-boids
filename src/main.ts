import './index.css';
import { BoidSimulation } from './sim/boidSimulation';
import { JevDirector } from './services/jevDirector';
import { HudController } from './ui/hud';

async function bootstrap() {
  const canvasContainer = document.getElementById('canvas-container')!;
  const hudRoot = document.getElementById('hud-root')!;

  const canvas = document.createElement('canvas');
  canvas.id = 'boid-canvas';
  canvasContainer.appendChild(canvas);

  // Initialize simulation
  const sim = new BoidSimulation({
    canvas,
    numParticles: 65536,
    onError: (msg) => {
      showErrorBanner(msg);
    },
  });

  const initialized = await sim.init();
  if (!initialized) {
    return;
  }

  // Initialize Jev Director
  const director = new JevDirector({
    uniformManager: sim.uniformManager,
    intervalMs: 600,
  });

  // Initialize HUD
  const hud = new HudController(hudRoot, sim, director);

  // Wire simulation telemetry to Jev cognitive loop and HUD
  sim['onTelemetry'] = (telemetry) => {
    director.ingestTelemetry(telemetry);
    hud.updateTelemetry(telemetry);
  };

  director['onDecision'] = (decision) => {
    hud.updateDecision(decision);
  };

  // Start GPU loop & Jev cognitive cycle
  sim.start();
  director.start();
}

function showErrorBanner(message: string) {
  const existing = document.querySelector('.webgpu-error-banner');
  if (existing) return;

  const banner = document.createElement('div');
  banner.className = 'webgpu-error-banner';
  banner.innerHTML = `
    <h2>WEBGPU ACCELERATION REQUIRED</h2>
    <p>${message}</p>
    <p style="margin-top: 14px; font-size: 0.8rem; color: #8b9bb4;">
      Please ensure you are using Google Chrome 113+, Microsoft Edge 113+, Safari 18+, or a WebGPU-enabled browser on a supported GPU.
    </p>
  `;
  document.body.appendChild(banner);
}

// Start application
bootstrap().catch((err) => {
  console.error('Bootstrap error:', err);
  showErrorBanner(err?.message || String(err));
});
