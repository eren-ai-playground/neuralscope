import { useEffect, useRef, useState } from 'react';
import type { Config, EpochMetric, Frame, FromWorker, Layer, Snapshot, ToWorker } from '../types';
import { describeChange, experimentKey, trainingProblem } from '../ml/config';
/** The last trained run before the current one, kept for the comparison overlay. */
export interface PreviousRun {
  history: EpochMetric[];
  change: string;
}
export function useExperiment(config: Config, reset: number, weights: Layer[] | null) {
  // The previous run's snapshot stays visible (marked pending) until the new worker answers, so panels do not flash to a loading state.
  const [latest, setLatest] = useState<{ id: number; snapshot: Snapshot } | null>(null);
  const [run, setRun] = useState(0),
    [error, setError] = useState('');
  const [previous, setPrevious] = useState<PreviousRun | null>(null),
    [frame, setFrame] = useState<Frame | null>(null);
  const worker = useRef<Worker | null>(null),
    runId = useRef(0);
  const last = useRef<{ snapshot: Snapshot; config: Config } | null>(null);
  // Timeline scrubbing asks for one frame at a time; newer requests replace queued ones.
  const frameBusy = useRef(false),
    queuedFrame = useRef<number | null>(null),
    replaying = useRef(false);
  const key = experimentKey(config),
    problem = trainingProblem(config);
  function sendFrame(epoch: number): void {
    if (!worker.current) return;
    if (frameBusy.current) {
      queuedFrame.current = epoch;
      return;
    }
    frameBusy.current = true;
    const m: ToWorker = { type: 'frame', runId: runId.current, epoch };
    worker.current.postMessage(m);
  }
  useEffect(() => {
    const id = ++runId.current;
    setRun(id);
    setError('');
    setFrame(null);
    frameBusy.current = false;
    queuedFrame.current = null;
    replaying.current = false;
    const before = last.current;
    if (before && before.snapshot.history.length > 1)
      setPrevious({ history: before.snapshot.history, change: describeChange(before.config, config) });
    if (problem) {
      worker.current = null;
      setLatest(null);
      return;
    }
    const runConfig = config;
    const w = new Worker(new URL('../workers/training.worker.ts', import.meta.url), { type: 'module' });
    worker.current = w;
    w.onmessage = (e: MessageEvent<FromWorker>) => {
      if (e.data.runId !== runId.current) return;
      if (e.data.type === 'error') setError(e.data.message);
      else if (e.data.type === 'frame') {
        frameBusy.current = false;
        if (replaying.current) setFrame(e.data.frame);
        if (queuedFrame.current !== null) {
          const epoch = queuedFrame.current;
          queuedFrame.current = null;
          sendFrame(epoch);
        }
      } else {
        setLatest({ id, snapshot: e.data.snapshot });
        last.current = { snapshot: e.data.snapshot, config: runConfig };
      }
    };
    w.onerror = e => setError(e.message || 'Worker başlatılamadı. Sayfayı yerel sunucudan açın.');
    const m: ToWorker = {
      type: 'init',
      runId: id,
      config,
      resolution: window.innerWidth < 650 ? 60 : 80,
      weights,
    };
    w.postMessage(m);
    return () => {
      w.terminate();
      if (worker.current === w) worker.current = null;
    };
    // Only a speed change preserves the trained model. All other changes reinitialize.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, reset, weights]);
  useEffect(() => {
    const m: ToWorker = { type: 'speed', runId: runId.current, speed: config.speed };
    worker.current?.postMessage(m);
  }, [config.speed]);
  const command = (type: 'start' | 'pause' | 'step') => {
    const m: ToWorker = { type, runId: runId.current };
    worker.current?.postMessage(m);
  };
  /** null returns to the live model. */
  const showFrame = (epoch: number | null) => {
    replaying.current = epoch !== null;
    if (epoch === null) {
      queuedFrame.current = null;
      setFrame(null);
    } else sendFrame(epoch);
  };
  return {
    snapshot: latest?.snapshot ?? null,
    pending: !problem && latest?.id !== run,
    error,
    problem,
    command,
    previous,
    clearPrevious: () => setPrevious(null),
    frame,
    showFrame,
  };
}
