import type { Config, EpochMetric, Layer, Model } from '../types';
import { validateConfig } from '../ml/config';
import { validateWeights } from '../ml/model';
export function download(name: string, data: string, type: string): void {
  const url = URL.createObjectURL(new Blob([data], { type })), a = document.createElement('a');
  a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function exportConfig(config: Config): void { download('neuralscope-deney.json', JSON.stringify({ schemaVersion: 1, config }, null, 2), 'application/json'); }
/** Settings plus the current weights. Optimizer moments are not saved. */
export function modelJSON(config: Config, model: Model, epoch: number): string {
  return JSON.stringify({ schemaVersion: 2, config, trainedEpochs: epoch, weights: model.layers });
}
export function exportModel(config: Config, model: Model, epoch: number): void { download('neuralscope-model.json', modelJSON(config, model, epoch), 'application/json'); }
/** Reads a settings file (v1), a model file with weights (v2) or a bare config object. */
export function importExperiment(text: string): { config: Config; weights: Layer[] | null; trainedEpochs: number | null } {
  let data: unknown;
  try { data = JSON.parse(text); } catch { throw Error('Dosya geçerli JSON içermiyor.'); }
  if (data && typeof data === 'object' && 'schemaVersion' in data) {
    if ((data.schemaVersion !== 1 && data.schemaVersion !== 2) || !('config' in data)) throw Error('Desteklenmeyen ayar dosyası sürümü.');
    const config = validateConfig(data.config);
    if (data.schemaVersion === 1 || !('weights' in data)) return { config, weights: null, trainedEpochs: null };
    const epochs = 'trainedEpochs' in data && Number.isInteger(data.trainedEpochs) ? data.trainedEpochs as number : null;
    return { config, weights: validateWeights(data.weights, config), trainedEpochs: epochs };
  }
  return { config: validateConfig(data), weights: null, trainedEpochs: null };
}
export function importConfig(text: string): Config { return importExperiment(text).config; }
export function historyCSV(history: EpochMetric[]): string {
  return 'epoch,train_bce,test_bce,train_accuracy,test_accuracy\r\n' + history.map(m => [m.epoch, m.trainLoss, m.testLoss ?? '', m.trainAccuracy, m.testAccuracy ?? ''].join(',')).join('\r\n');
}
