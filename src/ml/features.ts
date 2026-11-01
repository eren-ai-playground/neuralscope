import type { Feature } from '../types';
/** Input features the model can see. Order here is the canonical input order. */
export const FEATURES: Record<Feature, { label: string; value: (x1: number, x2: number) => number }> = {
  x1: { label: 'x₁', value: x1 => x1 },
  x2: { label: 'x₂', value: (_, x2) => x2 },
  x1sq: { label: 'x₁²', value: x1 => x1 * x1 },
  x2sq: { label: 'x₂²', value: (_, x2) => x2 * x2 },
  x1x2: { label: 'x₁·x₂', value: (x1, x2) => x1 * x2 },
  sin1: { label: 'sin(πx₁)', value: x1 => Math.sin(Math.PI * x1) },
  sin2: { label: 'sin(πx₂)', value: (_, x2) => Math.sin(Math.PI * x2) },
};
export const FEATURE_ORDER = Object.keys(FEATURES) as Feature[];
export const RAW_FEATURES: Feature[] = ['x1', 'x2'];
export function featurize(features: Feature[] | undefined, x: number[]): number[] {
  return features ? features.map(f => FEATURES[f].value(x[0], x[1])) : [...x];
}
