import type { Activation } from '../types';
export function sigmoid(x: number): number {
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}
export function activate(z: number, name: Activation): number {
  return name === 'relu' ? Math.max(0, z) : name === 'tanh' ? Math.tanh(z) : sigmoid(z);
}
export function derivative(z: number, a: number, name: Activation): number {
  return name === 'relu' ? (z > 0 ? 1 : 0) : name === 'tanh' ? 1 - a * a : a * (1 - a);
}
/** BCE from logits avoids log(0) and retains correct derivatives at saturation. */
export function binaryCrossEntropy(logit: number, target: number): number {
  return Math.max(logit, 0) - logit * target + Math.log1p(Math.exp(-Math.abs(logit)));
}
