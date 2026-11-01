import { describe, expect, it } from 'vitest';
import { activate, binaryCrossEntropy, sigmoid } from '../src/ml/activations';
import { DEFAULT, PRESETS, validateConfig } from '../src/ml/config';
import { generateData, splitData } from '../src/ml/datasets';
import {
  backward,
  batchGradient,
  createModel,
  evaluate,
  forward,
  objective,
  assertFinite,
} from '../src/ml/model';
import { TrainingSession } from '../src/ml/trainer';
import { ParameterOptimizer } from '../src/ml/optimizer';
import type { Activation, Model } from '../src/types';

describe('Sayısal doğruluk', () => {
  it('elle doğrulanabilen ileri geçişi hesaplar', () => {
    const model: Model = {
      activation: 'tanh',
      layers: [
        {
          w: [
            [0.5, -0.25],
            [0.3, 0.7],
          ],
          b: [0.1, -0.2],
        },
        { w: [[0.6, -0.8]], b: [0.05] },
      ],
    };
    const t = forward(model, [0.4, -0.6]);
    expect(t.z[0][0]).toBeCloseTo(0.45, 12);
    expect(t.z[0][1]).toBeCloseTo(-0.5, 12);
    const output = sigmoid(0.6 * Math.tanh(0.45) - 0.8 * Math.tanh(-0.5) + 0.05);
    expect(t.a[2][0]).toBeCloseTo(output, 12);
  });
  for (const activation of ['tanh', 'relu', 'sigmoid'] as Activation[]) {
    it(`${activation}: tüm ağırlık ve bias gradyanları finite difference ile eşleşir (h=1e-5, hata<2e-6)`, () => {
      const model = createModel({ ...DEFAULT, activation, hidden: [3, 2] });
      model.layers.forEach(l => l.b.fill(0.17)); // keep ReLU away from nondifferentiable zero
      const data = [
        { x: [0.35, -0.7] as [number, number], y: 1, id: 0 },
        { x: [-0.6, 0.12] as [number, number], y: 0, id: 1 },
      ];
      const l2 = 0.03,
        grad = batchGradient(model, data, l2),
        h = 1e-5;
      model.layers.forEach((l, k) => {
        const check = (array: number[], index: number, expected: number) => {
          const old = array[index];
          array[index] = old + h;
          const plus = objective(model, data, l2);
          array[index] = old - h;
          const minus = objective(model, data, l2);
          array[index] = old;
          expect(Math.abs((plus - minus) / (2 * h) - expected)).toBeLessThan(2e-6);
        };
        l.b.forEach((_, j) => {
          check(l.b, j, grad[k].b[j]);
          l.w[j].forEach((_, i) => check(l.w[j], i, grad[k].w[j][i]));
        });
      });
    });
  }
  it('uç sigmoid ve logit değerlerinde loss ve gradyan sonludur', () => {
    for (const z of [-10000, -1000, 0, 1000, 10000]) {
      expect(Number.isFinite(sigmoid(z))).toBe(true);
      for (const y of [0, 1]) expect(Number.isFinite(binaryCrossEntropy(z, y))).toBe(true);
    }
    expect(sigmoid(0)).toBe(0.5);
    expect(activate(-3, 'relu')).toBe(0);
    const m: Model = { activation: 'tanh', layers: [{ w: [[1000, 1000]], b: [1000] }] };
    expect(backward(m, { x: [1, 1], y: 0, id: 0 }).dz[0][0]).toBe(1);
    expect(() => assertFinite([{ w: [[Infinity]], b: [0] }], 'Test')).toThrow();
  });
  it('doğruluk eşiğini 0.5 olarak uygular ve tam küme ortalamasını alır', () => {
    const m: Model = { activation: 'tanh', layers: [{ w: [[0, 0]], b: [0] }] };
    const metrics = evaluate(m, [
      { x: [0, 0], y: 1, id: 0 },
      { x: [1, 1], y: 0, id: 1 },
    ]);
    expect(metrics.accuracy).toBe(0.5);
    expect(metrics.loss).toBeCloseTo(Math.log(2), 12);
  });
});

describe('Tekrarlanabilir deney ve eğitim', () => {
  for (const dataset of ['xor', 'circles', 'moons', 'spiral', 'linear'] as const) {
    it(`${dataset}: seed, veri ve bölme deterministiktir; kümeler ayrıdır`, () => {
      const config = { ...DEFAULT, dataset };
      const a = generateData(config);
      expect(a).toEqual(generateData(config));
      expect(a).not.toEqual(generateData({ ...config, seed: 43 }));
      const { train, test } = splitData(a, config),
        ids = new Set(train.map(p => p.id));
      expect(train.length + test.length).toBe(a.length);
      expect(test.every(p => !ids.has(p.id))).toBe(true);
      expect(splitData(a, { ...config, split: false }).test).toHaveLength(0);
    });
  }
  it('aynı seed ile başlangıç ağırlıkları aynıdır, farklı seed farklıdır', () => {
    expect(createModel(DEFAULT)).toEqual(createModel(DEFAULT));
    expect(createModel(DEFAULT)).not.toEqual(createModel({ ...DEFAULT, seed: 43 }));
  });
  it('SGD yeterince küçük adımlarla loss düşürür', () => {
    const model = createModel(DEFAULT),
      data = generateData({ ...DEFAULT, dataset: 'linear' });
    const initial = objective(model, data),
      optimizer = new ParameterOptimizer(model, 'sgd');
    for (let i = 0; i < 30; i++) optimizer.update(model, batchGradient(model, data, 0), 0.1);
    expect(objective(model, data)).toBeLessThan(initial);
  });
  it('son eksik batch dahil epoch sayımı ve reset doğrudur', () => {
    const config = { ...DEFAULT, samples: 41, split: false, batchSize: 16, epochs: 2 };
    const s = new TrainingSession(config),
      initial = structuredClone(s.model);
    s.step();
    expect(s.epoch).toBe(0);
    expect(s.cursor).toBe(16);
    expect(s.model).not.toEqual(initial);
    s.step();
    s.step();
    expect(s.epoch).toBe(1);
    expect(s.cursor).toBe(0);
    expect(s.history[1].trainLoss).toBeCloseTo(evaluate(s.model, s.train).loss, 12);
    while (!s.done) s.step();
    expect(s.batches).toBe(6);
    expect(s.history.map(m => m.epoch)).toEqual([0, 1, 2]);
    expect(s.step()).toBe(false);
    const reset = new TrainingSession(config);
    expect(reset.model).toEqual(initial);
    expect(reset.batches).toBe(0);
    const replay = new TrainingSession(config);
    while (!replay.done) replay.step();
    expect(replay.model).toEqual(s.model);
    expect(replay.history).toEqual(s.history);
  });
  it('XOR preset gerçek eğitimle train ve test doğruluğu >=%95 olur', () => {
    const s = new TrainingSession(PRESETS[0].config);
    while (!s.done) s.step();
    const last = s.history.at(-1)!;
    expect(last.trainAccuracy).toBeGreaterThanOrEqual(0.95);
    expect(last.testAccuracy).toBeGreaterThanOrEqual(0.95);
    expect(last.trainLoss).toBeLessThan(0.1);
    console.log('XOR doğrulaması:', JSON.stringify(last));
  });
  it('ısı haritası gerçek model olasılıklarını döndürür', () => {
    const s = new TrainingSession(DEFAULT),
      grid = s.grid(40);
    expect(grid).toHaveLength(1600);
    expect(grid.every(v => Number.isFinite(v) && v >= 0 && v <= 1)).toBe(true);
    expect(grid[0]).toBe(forward(s.model, [-1.5 + 1.5 / 40, 1.5 - 1.5 / 40]).a.at(-1)![0]);
  });
  it('bozuk parametreleri çalıştırmadan reddeder', () => {
    for (const patch of [
      { learningRate: NaN },
      { samples: 0 },
      { batchSize: 0 },
      { epochs: 0 },
      { hidden: [1] },
      { hidden: [2, 2, 2, 2] },
      { noise: -1 },
      { seed: -1 },
      { activation: 'bad' },
      { split: 'yes' },
    ]) {
      expect(() => validateConfig({ ...DEFAULT, ...patch })).toThrow();
    }
    expect(() => batchGradient(createModel(DEFAULT), [], 0)).toThrow();
  });
});
