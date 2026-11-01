import type { Activation } from '../types';
/** Probability colors shared by the decision surface and the output map: class 0 → class 1. */
export const PROBABILITY_COLORS: [number[], number[]] = [
  [36, 98, 117],
  [128, 88, 49],
];
// Activation maps follow the network panel: positive teal, negative orange, zero dark.
const ZERO = [17, 27, 33],
  POSITIVE = [114, 203, 213],
  NEGATIVE = [225, 166, 108];
function image(n: number, color: (i: number) => number[]): string {
  const canvas = document.createElement('canvas');
  canvas.width = n;
  canvas.height = n;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';
  const img = ctx.createImageData(n, n);
  for (let i = 0; i < n * n; i++) {
    const c = color(i);
    img.data.set([c[0], c[1], c[2], 255], i * 4);
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL();
}
const mix = (a: number[], b: number[], t: number) => a.map((v, k) => v + (b[k] - v) * t);
export function probabilityImage(values: number[], n: number): string {
  return image(n, i => mix(PROBABILITY_COLORS[0], PROBABILITY_COLORS[1], values[i]));
}
/**
 * Bounded activations (tanh, sigmoid) keep their true scale so saturation is visible.
 * Unbounded ones (ReLU, raw input features) are scaled by their largest magnitude on the plane.
 */
export function activationImage(values: number[], n: number, kind: Activation | 'input'): string {
  const range = kind === 'tanh' || kind === 'sigmoid' ? 1 : Math.max(1e-9, ...values.map(Math.abs));
  return image(n, i => {
    const t = Math.max(-1, Math.min(1, values[i] / range));
    return mix(ZERO, t >= 0 ? POSITIVE : NEGATIVE, Math.abs(t));
  });
}
