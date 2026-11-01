import type { FromWorker, Speed, Status, ToWorker } from '../types';
import { TrainingSession } from '../ml/trainer';
let session: TrainingSession | null = null, runId = 0, status: Status = 'ready';
let timer: ReturnType<typeof setTimeout> | undefined, speed: Speed = 'normal', resolution = 64;
let elapsed = 0, started = 0, lastSent = 0;
const MAP_RESOLUTION = 24;
const send = (message: FromWorker) => self.postMessage(message);
function snapshot(): void {
  if (!session) return;
  send({ type: 'snapshot', runId, snapshot: { model: session.model, epoch: session.epoch, batches: session.batches,
    epochProgress: session.cursor / session.train.length, history: session.history, elapsedMs: elapsed + (status === 'running' ? performance.now() - started : 0), status,
    grid: session.grid(resolution), resolution, sample: session.sample, batchGradient: session.lastGradient, update: session.lastUpdate,
    neuronMaps: session.neuronMaps(MAP_RESOLUTION), frameEpochs: session.frames.map(f => f.epoch) } });
  lastSent = performance.now();
}
function stop(next: Status): void {
  clearTimeout(timer); timer = undefined;
  if (status === 'running') elapsed += performance.now() - started;
  status = next;
}
function fail(error: unknown): void {
  stop('error'); send({ type: 'error', runId, message: error instanceof Error ? error.message : 'Eğitim sırasında beklenmeyen hata.' });
}
function tick(): void {
  if (!session || status !== 'running') return;
  try {
    const end = performance.now() + (speed === 'fast' ? 8 : speed === 'normal' ? 4 : 0);
    do { session.step(); } while (!session.done && speed !== 'slow' && performance.now() < end);
    if (session.done) { stop('done'); snapshot(); return; }
    if (performance.now() - lastSent >= 100) snapshot();
    timer = setTimeout(tick, speed === 'slow' ? 140 : speed === 'normal' ? 20 : 0);
  } catch (e) { fail(e); }
}
self.onmessage = (event: MessageEvent<ToWorker>) => {
  const m = event.data;
  try {
    if (m.type === 'init') {
      stop('ready'); runId = m.runId; session = new TrainingSession(m.config, m.weights);
      speed = m.config.speed; resolution = Math.max(40, Math.min(100, Math.floor(m.resolution)));
      elapsed = 0; snapshot(); return;
    }
    if (m.runId !== runId || !session || status === 'error') return;
    if (m.type === 'speed') { speed = m.speed; return; }
    if (m.type === 'frame') {
      const { epoch, model } = session.modelAt(m.epoch);
      send({ type: 'frame', runId, frame: { epoch, model, grid: session.grid(resolution, model), resolution, neuronMaps: session.neuronMaps(MAP_RESOLUTION, model) } }); return;
    }
    if (m.type === 'start' && status !== 'running' && !session.done) { status = 'running'; started = performance.now(); snapshot(); tick(); }
    if (m.type === 'pause') { stop(session.done ? 'done' : 'paused'); snapshot(); }
    if (m.type === 'step' && status !== 'running' && !session.done) {
      const t = performance.now(); session.step(); elapsed += performance.now() - t;
      status = session.done ? 'done' : 'paused'; snapshot();
    }
  } catch (e) { fail(e); }
};
