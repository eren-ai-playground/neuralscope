import { describe, expect, it } from 'vitest';
import { DEFAULT, describeChange, experimentKey, trainingProblem, validateConfig } from '../src/ml/config';
import { generateData } from '../src/ml/datasets';
import { featurize } from '../src/ml/features';
import { batchGradient, createModel, forward, objective } from '../src/ml/model';
import { TrainingSession } from '../src/ml/trainer';
import { importConfig, importExperiment, modelJSON } from '../src/utils/files';
import type { Config, CustomPoint } from '../src/types';

const square = (n: number): CustomPoint[] =>
  Array.from({ length: n }, (_, i) => ({
    x: [((i * 37) % 100) / 50 - 1, ((i * 61) % 100) / 50 - 1],
    y: i % 2,
  }));

describe('Girdi özellikleri ve gizli katmansız model', () => {
  it('özellikleri kanonik sırayla hesaplar; eski dosyalar ham x₁, x₂ ile açılır', () => {
    expect(featurize(['x1', 'x2', 'x1sq', 'x2sq', 'x1x2', 'sin1', 'sin2'], [0.5, -2])).toEqual([
      0.5,
      -2,
      0.25,
      4,
      -1,
      1,
      -Math.sin(2 * Math.PI),
    ]);
    expect(featurize(undefined, [0.3, 0.4])).toEqual([0.3, 0.4]);
    const { features: _features, customPoints: _points, ...old } = DEFAULT;
    expect(validateConfig(old)).toEqual(DEFAULT);
    expect(validateConfig({ ...DEFAULT, features: ['x1sq', 'x1'] }).features).toEqual(['x1', 'x1sq']);
    for (const features of [[], ['x3'], ['x1', 'x1'], 'x1'])
      expect(() => validateConfig({ ...DEFAULT, features })).toThrow();
  });
  for (const hidden of [[], [3]]) {
    it(`gizli=${JSON.stringify(hidden)}: ek özelliklerle tüm gradyanlar finite difference ile eşleşir`, () => {
      const config: Config = { ...DEFAULT, hidden, features: ['x1', 'x2sq', 'x1x2', 'sin1'] };
      const model = createModel(config);
      model.layers.forEach(l => l.b.fill(0.11));
      expect(model.layers[0].w[0]).toHaveLength(4);
      const data = [
        { x: [0.35, -0.7] as [number, number], y: 1, id: 0 },
        { x: [-0.6, 0.12] as [number, number], y: 0, id: 1 },
      ];
      const l2 = 0.02,
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
  it('halkalar x₁² ve x₂² ile gizli katman olmadan çözülür; ham girdilerle çözülemez', () => {
    const base: Config = { ...DEFAULT, dataset: 'circles', hidden: [], epochs: 200 };
    const train = (config: Config) => {
      const s = new TrainingSession(config);
      while (!s.done) s.step();
      return s.history.at(-1)!;
    };
    expect(train({ ...base, features: ['x1sq', 'x2sq'] }).testAccuracy).toBeGreaterThanOrEqual(0.97);
    expect(train(base).trainAccuracy).toBeLessThan(0.75);
  });
});

describe('Kendi çizimin', () => {
  it('noktaları gürültüsüz ve sırayla kullanır; doğrulama ve eğitim ön koşulu çalışır', () => {
    const points = square(10),
      config: Config = { ...DEFAULT, dataset: 'custom', customPoints: points, noise: 0.3 };
    expect(generateData(config).map(p => p.x)).toEqual(points.map(p => p.x));
    expect(trainingProblem(config)).toBeNull();
    expect(trainingProblem({ ...config, customPoints: points.slice(0, 3) })).toMatch(/en az 2/);
    expect(trainingProblem({ ...config, split: false, customPoints: points.slice(0, 2) })).toBeNull();
    expect(() => new TrainingSession({ ...config, customPoints: [] })).toThrow(/en az 2/);
    for (const bad of [[{ x: [2, 0], y: 0 }], [{ x: [0, 0], y: 2 }], [{ x: [0], y: 0 }], square(2001)])
      expect(() => validateConfig({ ...config, customPoints: bad })).toThrow();
  });
  it('çizilen noktalar yalnızca kendi çiziminde yeni deney sayılır', () => {
    expect(experimentKey({ ...DEFAULT, customPoints: square(4) })).toBe(experimentKey(DEFAULT));
    const custom: Config = { ...DEFAULT, dataset: 'custom', customPoints: square(4) };
    expect(experimentKey({ ...custom, customPoints: square(5) })).not.toBe(experimentKey(custom));
    expect(describeChange(DEFAULT, { ...DEFAULT, learningRate: 0.2, hidden: [] })).toBe(
      'Gizli katmanlar: 8-8 → yok · η: 0.02 → 0.2',
    );
    expect(describeChange(DEFAULT, DEFAULT)).toBe('Aynı ayarlar (sıfırlandı)');
  });
});

describe('Geri sarma kareleri ve nöron haritaları', () => {
  it('kareler sınırlı sayıda tutulur ve o epoch’taki modeli geri verir', () => {
    const s = new TrainingSession({ ...DEFAULT, epochs: 900 }),
      initial = structuredClone(s.model.layers);
    for (let i = 0; i < 300 && !s.done; i++) s.step();
    expect(s.frames.map(f => f.epoch).slice(0, 3)).toEqual([0, 3, 6]);
    expect(s.modelAt(0).model.layers).toEqual(initial);
    expect(s.modelAt(4).epoch).toBe(3);
    while (!s.done) s.step();
    expect(s.frames.length).toBeLessThanOrEqual(301);
    expect(s.frames.at(-1)!.epoch).toBe(900);
    expect(s.modelAt(900).model.layers).toEqual(s.model.layers);
    s.frames[0].layers[0].b[0] = 99;
    expect(s.modelAt(0).model.layers[0].b[0]).toBe(99);
    expect(s.model.layers[0].b[0]).not.toBe(99);
  });
  it('haritalar her giriş ve gizli nöronun gerçek aktivasyonudur', () => {
    const s = new TrainingSession({ ...DEFAULT, features: ['x1', 'x2', 'x1x2'], hidden: [3, 2] }),
      maps = s.neuronMaps(6);
    expect(maps.maps.map(layer => layer.length)).toEqual([3, 3, 2]);
    expect(maps.maps.flat().every(m => m.length === 36)).toBe(true);
    const t = forward(s.model, [-1.5 + 1.5 / 6, 1.5 - 1.5 / 6]);
    expect(maps.maps.map(layer => layer.map(m => m[0]))).toEqual([t.a[0], t.a[1], t.a[2]]);
    expect(new TrainingSession({ ...DEFAULT, hidden: [] }).neuronMaps(4).maps.map(l => l.length)).toEqual([
      2,
    ]);
  });
});

describe('Model dosyası', () => {
  it('ağırlıklarla kaydedilen model aynı ağırlıklarla başlar; eski ve bozuk dosyalar ayrılır', () => {
    const trained = new TrainingSession(DEFAULT);
    for (let i = 0; i < 40; i++) trained.step();
    const file = importExperiment(modelJSON(trained.config, trained.model, trained.epoch));
    expect(file.config).toEqual(DEFAULT);
    expect(file.trainedEpochs).toBe(trained.epoch);
    const resumed = new TrainingSession(file.config, file.weights);
    expect(resumed.model.layers).toEqual(trained.model.layers);
    expect(resumed.history[0].trainLoss).toBeCloseTo(trained.history.at(-1)!.trainLoss, 1);
    expect(importExperiment(JSON.stringify({ schemaVersion: 1, config: DEFAULT })).weights).toBeNull();
    expect(importConfig(modelJSON(DEFAULT, trained.model, 0))).toEqual(DEFAULT);
    const json = JSON.parse(modelJSON(DEFAULT, trained.model, 0));
    expect(() => importExperiment(JSON.stringify({ ...json, config: { ...DEFAULT, hidden: [4] } }))).toThrow(
      /şekli/,
    );
    json.weights[0].b[0] = 'x';
    expect(() => importExperiment(JSON.stringify(json))).toThrow(/sonlu/);
    expect(() => new TrainingSession(DEFAULT, [{ w: [[1]], b: [0] }])).toThrow(/şekli/);
  });
});
