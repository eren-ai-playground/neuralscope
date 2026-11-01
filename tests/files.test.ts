import { expect, it } from 'vitest';
import { DEFAULT } from '../src/ml/config';
import { historyCSV, importConfig } from '../src/utils/files';
it('JSON içe aktarma sürümü, tüm alanları ve geçersiz içeriği doğrular', () => {
  expect(importConfig(JSON.stringify({ schemaVersion: 1, config: DEFAULT }))).toEqual(DEFAULT);
  expect(importConfig(JSON.stringify(DEFAULT))).toEqual(DEFAULT);
  expect(() => importConfig('{bad')).toThrow();
  expect(() => importConfig(JSON.stringify({ schemaVersion: 9, config: DEFAULT }))).toThrow();
  expect(() => importConfig(JSON.stringify({ ...DEFAULT, learningRate: '0.02' }))).toThrow();
});
it('CSV gerçek epoch metriklerini saklar; olmayan test ölçümünü boş bırakır', () => {
  expect(historyCSV([{ epoch: 0, trainLoss: 0.71, testLoss: null, trainAccuracy: 0.5, testAccuracy: null }])).toBe('epoch,train_bce,test_bce,train_accuracy,test_accuracy\r\n0,0.71,,0.5,');
});
