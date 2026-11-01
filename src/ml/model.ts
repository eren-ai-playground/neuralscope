import type { Config, Gradients, Layer, Metrics, Model, Point, Trace } from '../types';
import { activate, binaryCrossEntropy, derivative, sigmoid } from './activations';
import { Random } from './random';
import { featurize } from './features';
export function createModel(config: Config): Model {
  const rng = new Random(config.seed ^ 0xbad5eed), sizes = modelSizes(config);
  return { activation: config.activation, features: [...config.features], layers: sizes.slice(1).map((out, l) => ({
    w: Array.from({ length: out }, () => Array.from({ length: sizes[l] }, () => {
      const scale = config.activation === 'relu' && l < sizes.length - 2 ? Math.sqrt(2 / sizes[l]) : Math.sqrt(2 / (sizes[l] + out));
      return rng.normal() * scale;
    })), b: Array(out).fill(0),
  })) };
}
/** Layer widths: selected input features, hidden layers, one sigmoid output. */
export function modelSizes(config: Config): number[] { return [config.features.length, ...config.hidden, 1]; }
/** Accepts saved weights only if every shape matches the configuration and every value is a usable number. */
export function validateWeights(value: unknown, config: Config): Layer[] {
  const sizes = modelSizes(config), layers = value as Layer[];
  const ok = Array.isArray(layers) && layers.length === sizes.length - 1 && layers.every((l, k) => l && Array.isArray(l.b) && Array.isArray(l.w)
    && l.b.length === sizes[k + 1] && l.w.length === sizes[k + 1] && l.w.every(row => Array.isArray(row) && row.length === sizes[k]));
  if (!ok) throw Error(`Ağırlıkların şekli ayarlarla uyuşmuyor (${sizes.join(" → ")} bekleniyor).`);
  if ([...layers.flatMap(l => l.b), ...layers.flatMap(l => l.w.flat())].some(v => typeof v !== "number" || !Number.isFinite(v) || Math.abs(v) > 1e6)) throw Error("Ağırlıklar sonlu ve mutlak değeri 1e6’dan küçük sayılar olmalı.");
  return copyLayers(layers);
}
export function copyLayers(layers: Layer[]): Layer[] { return layers.map(l => ({ w: l.w.map(row => [...row]), b: [...l.b] })); }
export function zeros(model: Model): Layer[] {
  return model.layers.map(l => ({ w: l.w.map(row => row.map(() => 0)), b: l.b.map(() => 0) }));
}
export function forward(model: Model, x: number[]): Trace {
  const a: number[][] = [featurize(model.features, x)], z: number[][] = [];
  model.layers.forEach((layer, l) => {
    const values = layer.w.map((row, j) => row.reduce((sum, w, i) => sum + w * a[l][i], layer.b[j]));
    z.push(values);
    a.push(values.map(v => l === model.layers.length - 1 ? sigmoid(v) : activate(v, model.activation)));
  });
  return { a, z };
}
export function backward(model: Model, point: Point, trace = forward(model, point.x)): Gradients {
  const layers = zeros(model), dz: number[][] = model.layers.map(l => l.b.map(() => 0));
  const last = model.layers.length - 1;
  dz[last][0] = trace.a[last + 1][0] - point.y; // sigmoid + BCE simplifies exactly to p-y
  for (let l = last; l >= 0; l--) {
    model.layers[l].b.forEach((_, j) => {
      layers[l].b[j] = dz[l][j];
      layers[l].w[j] = trace.a[l].map(a => dz[l][j] * a);
    });
    if (l > 0) {
      dz[l - 1] = model.layers[l - 1].b.map((_, i) =>
        model.layers[l].w.reduce((sum, row, j) => sum + row[i] * dz[l][j], 0) * derivative(trace.z[l - 1][i], trace.a[l][i], model.activation));
    }
  }
  return { layers, dz };
}
export function batchGradient(model: Model, batch: Point[], l2: number): Layer[] {
  if (!batch.length) throw Error('Boş batch ile eğitim yapılamaz.');
  const result = zeros(model);
  for (const point of batch) {
    const g = backward(model, point);
    result.forEach((l, k) => {
      l.b.forEach((_, j) => { l.b[j] += g.layers[k].b[j] / batch.length;
        l.w[j].forEach((_, i) => { l.w[j][i] += g.layers[k].w[j][i] / batch.length; });
      });
    });
  }
  result.forEach((l, k) => l.w.forEach((row, j) => row.forEach((_, i) => { row[i] += l2 * model.layers[k].w[j][i]; })));
  assertFinite(result, 'Gradyan');
  return result;
}
export function evaluate(model: Model, data: Point[]): Metrics {
  if (!data.length) throw Error('Ölçüm için veri yok.');
  let loss = 0, correct = 0;
  for (const p of data) {
    const t = forward(model, p.x), logit = t.z[t.z.length - 1][0];
    loss += binaryCrossEntropy(logit, p.y);
    if (Number(t.a[t.a.length - 1][0] >= 0.5) === p.y) correct++;
  }
  return { loss: loss / data.length, accuracy: correct / data.length };
}
export function objective(model: Model, data: Point[], l2 = 0): number {
  return evaluate(model, data).loss + 0.5 * l2 * model.layers.reduce((s, l) => s + l.w.flat().reduce((t, w) => t + w * w, 0), 0);
}
export function assertFinite(layers: Layer[], name: string): void {
  for (const l of layers) for (const v of [...l.b, ...l.w.flat()]) {
    if (!Number.isFinite(v) || Math.abs(v) > 1e6) throw Error(`${name} sayısal sınırı aştı. Öğrenme hızını azaltıp deneyi sıfırlayın.`);
  }
}
