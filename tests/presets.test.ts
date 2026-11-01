import { expect, it } from 'vitest';
import { PRESETS } from '../src/ml/config';
import { TrainingSession } from '../src/ml/trainer';
for (const preset of PRESETS.slice(1)) {
  it(`${preset.name}: sabit seed ile gerçek eğitim ve test genellemesi`, () => {
    const session = new TrainingSession(preset.config);
    while (!session.done) session.step();
    const last = session.history.at(-1)!;
    console.log(preset.name, JSON.stringify(last));
    expect(last.trainAccuracy).toBeGreaterThanOrEqual(0.94);
    expect(last.testAccuracy).toBeGreaterThanOrEqual(0.9);
    expect(last.trainLoss).toBeLessThan(session.history[0].trainLoss);
  });
}
