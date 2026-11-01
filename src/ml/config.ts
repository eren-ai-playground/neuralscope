import type { Config, CustomPoint, Feature } from '../types';
import { FEATURES, FEATURE_ORDER } from './features';
export const DEFAULT: Config = { dataset: 'xor', samples: 300, noise: 0.04, seed: 42, split: true, features: ['x1', 'x2'], hidden: [8, 8], activation: 'tanh', optimizer: 'adam', learningRate: 0.02, batchSize: 32, epochs: 400, l2: 0, speed: 'normal', customPoints: [] };
export const PRESETS: { name: string; description: string; config: Config }[] = [
  { name: 'XOR öğren', description: 'Doğrusal olmayan ilk problem', config: { ...DEFAULT } },
  { name: 'Ayları ayır', description: 'İki kıvrımı birbirinden ayır', config: { ...DEFAULT, dataset: 'moons', noise: 0.06, hidden: [8, 8], epochs: 350 } },
  { name: 'Halkaları çöz', description: 'İçerisi ve dışarısı', config: { ...DEFAULT, dataset: 'circles', noise: 0.04, epochs: 400 } },
  { name: 'Spiral deneyi', description: 'Kapasitenin sınırlarını keşfet', config: { ...DEFAULT, dataset: 'spiral', samples: 500, noise: 0.025, hidden: [16, 16, 16], epochs: 1200, learningRate: 0.01 } },
];
export const DATASETS = { xor: 'XOR', circles: 'İç içe halkalar', moons: 'İki ay', spiral: 'İki kollu spiral', linear: 'Doğrusal', custom: 'Kendi çizimin' };
export const MAX_POINTS = 2000;
/** Every setting except observation speed defines a new experiment. Drawn points only matter for the custom dataset. */
export const experimentKey = (c: Config): string => JSON.stringify({ ...c, speed: undefined, customPoints: c.dataset === 'custom' ? c.customPoints : undefined });
/** Why this configuration cannot be trained yet, or null. Only drawn data can be incomplete. */
export function trainingProblem(c: Config): string | null {
  if (c.dataset !== 'custom') return null;
  const need = c.split ? 2 : 1, counts = [0, 1].map(y => c.customPoints.filter(p => p.y === y).length);
  return counts.every(n => n >= need) ? null : `Kendi verinde her sınıftan en az ${need} nokta olmalı (şu an sınıf 0: ${counts[0]}, sınıf 1: ${counts[1]}). Karar sınırına tıklayarak nokta ekle.`;
}
export function validateConfig(value: unknown): Config {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw Error('Ayar dosyası bir JSON nesnesi olmalı.');
  const c = value as Config;
  const number = (key: keyof Config, min: number, max: number, integer = false) => {
    const v = c[key];
    if (typeof v !== 'number' || !Number.isFinite(v) || v < min || v > max || (integer && !Number.isInteger(v))) throw Error(`${key}: ${min} ile ${max} arasında ${integer ? 'tam sayı' : 'sayı'} bekleniyor.`);
  };
  number('samples', 40, 2000, true); number('noise', 0, 0.35); number('seed', 0, 4294967295, true);
  number('learningRate', 0.0001, 3); number('batchSize', 1, 256, true); number('epochs', 1, 5000, true); number('l2', 0, 0.1);
  if (!Object.hasOwn(DATASETS, c.dataset)) throw Error('Bilinmeyen veri kümesi.');
  if (!['tanh', 'relu', 'sigmoid'].includes(c.activation)) throw Error('Bilinmeyen aktivasyon.');
  if (!['sgd', 'adam'].includes(c.optimizer)) throw Error('Bilinmeyen optimizer.');
  if (!['slow', 'normal', 'fast'].includes(c.speed)) throw Error('Bilinmeyen gözlem hızı.');
  if (typeof c.split !== 'boolean') throw Error('split boolean olmalı.');
  if (!Array.isArray(c.hidden) || c.hidden.length > 3 || c.hidden.some(n => !Number.isInteger(n) || n < 2 || n > 16)) throw Error('0–3 gizli katman ve katman başına 2–16 nöron gerekli.');
  // Files saved before input features and drawn data existed keep working with the old defaults.
  const features = c.features ?? DEFAULT.features, points = c.customPoints ?? [];
  if (!Array.isArray(features) || !features.length || features.some(f => !Object.hasOwn(FEATURES, f)) || new Set(features).size !== features.length) throw Error('En az bir geçerli ve tekrarsız girdi özelliği gerekli.');
  if (!Array.isArray(points) || points.length > MAX_POINTS || points.some(p => !p || !Array.isArray(p.x) || p.x.length !== 2 || p.x.some(v => typeof v !== 'number' || !Number.isFinite(v) || Math.abs(v) > 1.5) || (p.y !== 0 && p.y !== 1))) throw Error(`Çizilen noktalar en fazla ${MAX_POINTS} adet, koordinatları [-1.5, 1.5] içinde ve sınıfı 0 ya da 1 olmalı.`);
  return { dataset: c.dataset, samples: c.samples, noise: c.noise, seed: c.seed, split: c.split, features: FEATURE_ORDER.filter(f => features.includes(f)), hidden: [...c.hidden], activation: c.activation, optimizer: c.optimizer, learningRate: c.learningRate, batchSize: c.batchSize, epochs: c.epochs, l2: c.l2, speed: c.speed, customPoints: points.map((p): CustomPoint => ({ x: [p.x[0], p.x[1]], y: p.y })) };
}
const LABELS: Partial<Record<keyof Config, string>> = { dataset: 'Veri', samples: 'Örnek', noise: 'Gürültü', seed: 'Seed', split: 'Test bölmesi', features: 'Girdiler', hidden: 'Gizli katmanlar', activation: 'Aktivasyon', optimizer: 'Optimizer', learningRate: 'η', batchSize: 'Batch', epochs: 'Epoch', l2: 'L2', customPoints: 'Çizilen noktalar' };
const show = (key: keyof Config, c: Config): string => {
  const v = c[key];
  if (key === 'dataset') return DATASETS[c.dataset];
  if (key === 'features') return (v as Feature[]).map(f => FEATURES[f].label).join(', ');
  if (key === 'hidden') return (v as number[]).length ? (v as number[]).join('-') : 'yok';
  if (key === 'customPoints') return `${(v as CustomPoint[]).length} nokta`;
  if (key === 'split') return v ? 'açık' : 'kapalı';
  return String(v);
};
/** Human-readable list of what differs between two experiments, e.g. "η: 0.02 → 0.5". */
export function describeChange(from: Config, to: Config): string {
  const keys = (Object.keys(LABELS) as (keyof Config)[]).filter(k => (k !== 'customPoints' || to.dataset === 'custom') && JSON.stringify(from[k]) !== JSON.stringify(to[k]));
  return keys.length ? keys.map(k => `${LABELS[k]}: ${show(k, from)} → ${show(k, to)}`).join(' · ') : 'Aynı ayarlar (sıfırlandı)';
}
