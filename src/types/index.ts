export type Activation = 'tanh' | 'relu' | 'sigmoid';
export type DatasetName = 'xor' | 'circles' | 'moons' | 'spiral' | 'linear' | 'custom';
export type Feature = 'x1' | 'x2' | 'x1sq' | 'x2sq' | 'x1x2' | 'sin1' | 'sin2';
export type Optimizer = 'sgd' | 'adam';
export type Speed = 'slow' | 'normal' | 'fast';
export type Status = 'ready' | 'running' | 'paused' | 'done' | 'error';
export interface CustomPoint { x: [number, number]; y: number }
export interface Config {
  dataset: DatasetName; samples: number; noise: number; seed: number;
  split: boolean; features: Feature[]; hidden: number[]; activation: Activation;
  optimizer: Optimizer; learningRate: number; batchSize: number;
  epochs: number; l2: number; speed: Speed; customPoints: CustomPoint[];
}
export interface Point { x: [number, number]; y: number; id: number }
export interface Layer { w: number[][]; b: number[] }
/** features defaults to raw [x1, x2] when omitted. */
export interface Model { layers: Layer[]; activation: Activation; features?: Feature[] }
export interface Trace { a: number[][]; z: number[][] }
export interface Gradients { layers: Layer[]; dz: number[][] }
export interface Metrics { loss: number; accuracy: number }
export interface EpochMetric { epoch: number; trainLoss: number; testLoss: number | null; trainAccuracy: number; testAccuracy: number | null }
/** maps[k][j]: activation of neuron j in trace layer k (0 = inputs) over a resolution² grid, row 0 at the top. */
export interface NeuronMaps { resolution: number; maps: number[][][] }
export interface Snapshot {
  model: Model; epoch: number; batches: number; epochProgress: number;
  history: EpochMetric[]; elapsedMs: number; status: Status;
  grid: number[]; resolution: number; sample: Point;
  batchGradient: Layer[] | null; update: Layer[] | null;
  neuronMaps: NeuronMaps; frameEpochs: number[];
}
/** A stored past state used for timeline replay. */
export interface Frame { epoch: number; model: Model; grid: number[]; resolution: number; neuronMaps: NeuronMaps }
export type ToWorker =
  | { type: 'init'; runId: number; config: Config; resolution: number; weights: Layer[] | null }
  | { type: 'start' | 'pause' | 'step'; runId: number }
  | { type: 'speed'; runId: number; speed: Speed }
  | { type: 'frame'; runId: number; epoch: number };
export type FromWorker =
  | { type: 'snapshot'; runId: number; snapshot: Snapshot }
  | { type: 'frame'; runId: number; frame: Frame }
  | { type: 'error'; runId: number; message: string };
export interface Selection { layer: number; neuron: number; input?: number }
