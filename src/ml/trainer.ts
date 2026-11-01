import type { Config, EpochMetric, Layer, Model, NeuronMaps, Point } from '../types';
import { trainingProblem, validateConfig } from './config';
import { EXTENT, generateData, splitData } from './datasets';
import { batchGradient, copyLayers, createModel, evaluate, forward, validateWeights } from './model';
import { ParameterOptimizer } from './optimizer';
import { Random } from './random';
/** Timeline replay keeps at most about this many epoch states. */
export const MAX_FRAMES = 300;
export class TrainingSession {
  readonly config: Config;
  readonly model: Model;
  readonly train: Point[];
  readonly test: Point[];
  readonly history: EpochMetric[] = [];
  readonly frames: { epoch: number; layers: Layer[] }[] = [];
  epoch = 0;
  batches = 0;
  cursor = 0;
  sample: Point;
  lastGradient: Layer[] | null = null;
  lastUpdate: Layer[] | null = null;
  private order: Point[];
  private random: Random;
  private optimizer: ParameterOptimizer;
  private frameStride: number;
  /** weights replaces the seeded initialization; optimizer moments still start from zero. */
  constructor(config: Config, weights: Layer[] | null = null) {
    this.config = validateConfig(config);
    const problem = trainingProblem(this.config);
    if (problem) throw Error(problem);
    this.model = createModel(this.config);
    if (weights) this.model.layers = validateWeights(weights, this.config);
    const data = splitData(generateData(this.config), this.config);
    this.train = data.train;
    this.test = data.test;
    this.random = new Random(this.config.seed ^ 0xc0ffee);
    this.order = this.random.shuffle([...this.train]);
    this.sample = this.order[0];
    this.frameStride = Math.max(1, Math.ceil(this.config.epochs / MAX_FRAMES));
    this.optimizer = new ParameterOptimizer(this.model, this.config.optimizer);
    this.measure();
  }
  get done(): boolean {
    return this.epoch >= this.config.epochs;
  }
  step(): boolean {
    if (this.done) return false;
    const batch = this.order.slice(this.cursor, this.cursor + this.config.batchSize);
    this.lastGradient = batchGradient(this.model, batch, this.config.l2);
    this.lastUpdate = this.optimizer.update(this.model, this.lastGradient, this.config.learningRate);
    this.sample = batch[batch.length - 1];
    this.batches++;
    this.cursor += batch.length;
    if (this.cursor === this.train.length) {
      this.epoch++;
      this.measure();
      this.cursor = 0;
      if (!this.done) this.order = this.random.shuffle([...this.train]);
    }
    return true;
  }
  private measure(): void {
    const train = evaluate(this.model, this.train),
      test = this.test.length ? evaluate(this.model, this.test) : null;
    this.history.push({
      epoch: this.epoch,
      trainLoss: train.loss,
      testLoss: test?.loss ?? null,
      trainAccuracy: train.accuracy,
      testAccuracy: test?.accuracy ?? null,
    });
    if (!Number.isFinite(train.loss) || (test && !Number.isFinite(test.loss)))
      throw Error('Loss sonlu değil; deneyi sıfırlayın.');
    if (this.epoch % this.frameStride === 0 || this.done)
      this.frames.push({ epoch: this.epoch, layers: copyLayers(this.model.layers) });
  }
  /** The model as it was at a recorded epoch (the nearest earlier frame). */
  modelAt(epoch: number): { epoch: number; model: Model } {
    const frame = [...this.frames].reverse().find(f => f.epoch <= epoch) ?? this.frames[0];
    return { epoch: frame.epoch, model: { ...this.model, layers: copyLayers(frame.layers) } };
  }
  grid(resolution: number, model: Model = this.model): number[] {
    const result: number[] = [];
    for (let y = 0; y < resolution; y++)
      for (let x = 0; x < resolution; x++) {
        const t = forward(model, cell(x, y, resolution));
        result.push(t.a[t.a.length - 1][0]);
      }
    return result;
  }
  /** Activation of every input and hidden neuron over the input plane; the output map is the main grid. */
  neuronMaps(resolution: number, model: Model = this.model): NeuronMaps {
    const maps = model.layers.map((l, k) =>
      Array.from({ length: k === 0 ? l.w[0].length : model.layers[k - 1].b.length }, () => [] as number[]),
    );
    for (let y = 0; y < resolution; y++)
      for (let x = 0; x < resolution; x++) {
        const t = forward(model, cell(x, y, resolution));
        maps.forEach((layer, k) => layer.forEach((map, j) => map.push(t.a[k][j])));
      }
    return { resolution, maps };
  }
}
/** Center of grid cell (x, y); row 0 is the top (x₂ = +EXTENT). */
const cell = (x: number, y: number, n: number): [number, number] => [
  -EXTENT + ((x + 0.5) / n) * EXTENT * 2,
  EXTENT - ((y + 0.5) / n) * EXTENT * 2,
];
