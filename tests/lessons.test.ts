import { describe, expect, it } from 'vitest';
import { DEFAULT } from '../src/ml/config';
import { LESSONS, stepConfig } from '../src/ml/lessons';
import { TrainingSession } from '../src/ml/trainer';
import type { Config, EpochMetric } from '../src/types';

const train = (config: Config): EpochMetric[] => {
  const session = new TrainingSession(config);
  while (!session.done) session.step();
  return session.history;
};

describe('Rehberli dersler', () => {
  it('ders kimlikleri benzersizdir ve her dersin en az iki adımı vardır', () => {
    expect(new Set(LESSONS.map(l => l.id)).size).toBe(LESSONS.length);
    for (const lesson of LESSONS) expect(lesson.steps.length).toBeGreaterThanOrEqual(2);
  });
  for (const lesson of LESSONS)
    for (const step of lesson.steps)
      it(`${lesson.title} / ${step.title}: anlatılan sonuç gerçek eğitimde çıkar`, () => {
        const outcome = step.explain(train(stepConfig(lesson, step, DEFAULT)));
        expect(outcome.text).not.toMatch(/NaN|undefined|Infinity/);
        expect(outcome.expected, outcome.text).toBe(true);
      });
  it('kapasite dersindeki üç nöron iddiası ölçümle uyuşur', () => {
    const lesson = LESSONS.find(l => l.id === 'capacity')!,
      base = stepConfig(lesson, lesson.steps[0], DEFAULT);
    const accuracy = (seed: number) => train({ ...base, hidden: [3], seed }).at(-1)!.trainAccuracy;
    expect(accuracy(42)).toBeLessThan(0.95);
    for (const seed of [1, 2, 3, 7]) expect(accuracy(seed)).toBeGreaterThanOrEqual(0.98);
  });
  it('beklenmeyen sonuçta başarı anlatmaz, ölçülen değeri söyler', () => {
    const step = LESSONS.find(l => l.id === 'capacity')!.steps[2];
    const failed = [{ epoch: 1, trainLoss: 0.6, testLoss: 0.7, trainAccuracy: 0.6, testAccuracy: 0.55 }];
    const outcome = step.explain(failed);
    expect(outcome.expected).toBe(false);
    expect(outcome.text).toContain('%60.0');
    expect(outcome.text).toContain('%55.0');
  });
  it('dersler çizilen noktaları silmez ve geçerli ayar üretir', () => {
    const customPoints = [{ x: [0.1, 0.2] as [number, number], y: 1 }];
    for (const lesson of LESSONS)
      for (const step of lesson.steps)
        expect(stepConfig(lesson, step, { customPoints }).customPoints).toEqual(customPoints);
  });
});
