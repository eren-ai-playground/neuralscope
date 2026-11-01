import type { Config, Point } from '../types';
import { Random } from './random';
export const EXTENT = 1.5;
/** Dataset coordinates are already model coordinates; no hidden scaling at inference. */
export function generateData(config: Config): Point[] {
  // Drawn points are used exactly as placed: no sampling and no noise.
  if (config.dataset === 'custom')
    return config.customPoints.map((p, id) => ({ x: [p.x[0], p.x[1]], y: p.y, id }));
  const rng = new Random(config.seed);
  const result: Point[] = [];
  for (let i = 0; i < config.samples; i++) {
    let x1: number, x2: number;
    let y = i % 2;
    const angle = rng.next() * Math.PI * 2;
    switch (config.dataset) {
      case 'xor':
        x1 = (rng.next() * 0.8 + 0.2) * (rng.next() < 0.5 ? -1 : 1);
        x2 = (rng.next() * 0.8 + 0.2) * (rng.next() < 0.5 ? -1 : 1);
        y = x1 * x2 > 0 ? 0 : 1;
        break;
      case 'circles': {
        const r = y ? 0.48 : 1.05;
        x1 = r * Math.cos(angle);
        x2 = r * Math.sin(angle);
        break;
      }
      case 'moons': {
        const t = rng.next() * Math.PI;
        x1 = (y ? 1 - Math.cos(t) : Math.cos(t)) * 0.75 - 0.375;
        x2 = (y ? 0.45 - Math.sin(t) : Math.sin(t)) * 0.9 - 0.2;
        break;
      }
      case 'spiral': {
        const r = 0.1 + rng.next() * 1.1,
          t = r * 4.7 + y * Math.PI;
        x1 = r * Math.cos(t);
        x2 = r * Math.sin(t);
        break;
      }
      case 'linear':
        x1 = rng.next() * 2.2 - 1.1;
        x2 = rng.next() * 2.2 - 1.1;
        y = x2 > 0.5 * x1 ? 1 : 0;
        break;
    }
    // Noise perturbs positions, keeping original target labels unchanged.
    x1 += rng.normal() * config.noise;
    x2 += rng.normal() * config.noise;
    result.push({ x: [x1, x2], y, id: i });
  }
  return result;
}
export function splitData(data: Point[], config: Config): { train: Point[]; test: Point[] } {
  if (!config.split) return { train: [...data], test: [] };
  // Stratification preserves both classes; held-out points never enter a batch.
  const rng = new Random(config.seed ^ 0x715ab);
  const train: Point[] = [],
    test: Point[] = [];
  for (const y of [0, 1]) {
    const group = rng.shuffle(data.filter(p => p.y === y));
    const cut = Math.floor(group.length * 0.8);
    train.push(...group.slice(0, cut));
    test.push(...group.slice(cut));
  }
  return { train: rng.shuffle(train), test: rng.shuffle(test) };
}
