import type { Layer, Model, Optimizer } from '../types';
import { assertFinite, zeros } from './model';
export class ParameterOptimizer {
  private m: Layer[];
  private v: Layer[];
  private t = 0;
  constructor(
    model: Model,
    private name: Optimizer,
  ) {
    this.m = zeros(model);
    this.v = zeros(model);
  }
  update(model: Model, gradient: Layer[], rate: number): Layer[] {
    this.t++;
    const delta = zeros(model);
    model.layers.forEach((layer, l) => {
      const change = (g: number, j: number, i?: number) => {
        if (this.name === 'sgd') return -rate * g;
        const m = i === undefined ? this.m[l].b : this.m[l].w[j];
        const v = i === undefined ? this.v[l].b : this.v[l].w[j];
        const index = i ?? j;
        m[index] = 0.9 * m[index] + 0.1 * g;
        v[index] = 0.999 * v[index] + 0.001 * g * g;
        return (
          (-rate * (m[index] / (1 - 0.9 ** this.t))) / (Math.sqrt(v[index] / (1 - 0.999 ** this.t)) + 1e-8)
        );
      };
      layer.b.forEach((_, j) => {
        delta[l].b[j] = change(gradient[l].b[j], j);
        layer.b[j] += delta[l].b[j];
        layer.w[j].forEach((_, i) => {
          delta[l].w[j][i] = change(gradient[l].w[j][i], j, i);
          layer.w[j][i] += delta[l].w[j][i];
        });
      });
    });
    assertFinite(model.layers, 'Model parametresi');
    return delta;
  }
}
