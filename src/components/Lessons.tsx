import { useState } from 'react';
import type { Config, Snapshot } from '../types';
import { experimentKey } from '../ml/config';
import { LESSONS, stepConfig } from '../ml/lessons';

const STORE = 'neuralscope.lessons.v1';
function loadCompleted(): string[] {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE) ?? '[]');
    return Array.isArray(saved) ? saved : [];
  } catch {
    return [];
  }
}

/** Guided experiments. The panel only sets up runs and reads their real results; training stays in the worker. */
export function Lessons({
  config,
  snapshot,
  pending,
  error,
  onApply,
  onTrain,
}: {
  config: Config;
  snapshot: Snapshot | null;
  pending: boolean;
  error: string;
  /** fresh: a lesson is starting, so the earlier run is not shown as its comparison. */
  onApply: (config: Config, fresh: boolean) => void;
  onTrain: (type: 'start' | 'pause') => void;
}) {
  const [active, setActive] = useState<{ lesson: number; step: number } | null>(null);
  const [completed, setCompleted] = useState(loadCompleted);
  const open = (lesson: number, step: number) => {
    setActive({ lesson, step });
    onApply(stepConfig(LESSONS[lesson], LESSONS[lesson].steps[step], config), step === 0);
  };
  const finish = (id: string) => {
    const next = completed.includes(id) ? completed : [...completed, id];
    setCompleted(next);
    setActive(null);
    try {
      localStorage.setItem(STORE, JSON.stringify(next));
    } catch {
      /* Progress marks are a convenience; lessons work without them. */
    }
  };

  if (!active)
    return (
      <section className="lessons" aria-label="Rehberli dersler">
        <div className="lessons-head">
          <span className="eyebrow accent">REHBERLİ DERSLER</span>
          <p>
            Bir kavramı adım adım dene: ayarları ders kurar, eğitimi sen başlatırsın, sonucu birlikte
            yorumlarız.
          </p>
        </div>
        <div className="lesson-cards">
          {LESSONS.map((lesson, i) => (
            <button key={lesson.id} type="button" className="lesson-card" onClick={() => open(i, 0)}>
              <strong>
                {completed.includes(lesson.id) && (
                  <span className="lesson-check" aria-label="Tamamlandı">
                    ✓{' '}
                  </span>
                )}
                {lesson.title}
              </strong>
              <span>{lesson.summary}</span>
              <small>{lesson.steps.length} adım</small>
            </button>
          ))}
        </div>
      </section>
    );

  const lesson = LESSONS[active.lesson],
    step = lesson.steps[active.step];
  const last = active.step === lesson.steps.length - 1;
  const matches = experimentKey(config) === experimentKey(stepConfig(lesson, step, config));
  const status = snapshot?.status;
  const finished = matches && !pending && (status === 'done' || !!error);
  const result = finished && !error && snapshot ? step.explain(snapshot.history) : null;
  return (
    <section className="lessons lesson-active" aria-label={`Ders: ${lesson.title}`}>
      <div className="lesson-top">
        <div>
          <span className="eyebrow accent">
            DERS · {lesson.title} · ADIM {active.step + 1} / {lesson.steps.length}
          </span>
          <h2>{step.title}</h2>
        </div>
        <button type="button" className="text-button" onClick={() => setActive(null)}>
          Dersten çık ×
        </button>
      </div>
      <p>{step.task}</p>
      <p className="lesson-look">👁 {step.look}</p>
      <div className="lesson-result" role="status">
        {!matches && (
          <p className="lesson-warning">
            Ayarları değiştirdin, bu yüzden bu adımın sonucu yorumlanmıyor. Dilediğin kadar dene; hazır olunca
            adımı yeniden kurabilirsin.
          </p>
        )}
        {matches && error && (
          <p className="lesson-warning">
            Eğitim sayısal sınırı aştığı için durdu. Bu da bir sonuçtur: adımlar ağırlıkları kontrolsüz
            büyüttü.
          </p>
        )}
        {result && <p className={result.expected ? '' : 'lesson-warning'}>{result.text}</p>}
        {result && last && <p className="lesson-conclusion">{lesson.conclusion}</p>}
        {matches && !finished && status === 'running' && (
          <p className="lesson-wait">
            Eğitim sürüyor: epoch {snapshot?.epoch} / {config.epochs}
          </p>
        )}
      </div>
      <div className="lesson-actions">
        {!matches ? (
          <button type="button" className="primary-button" onClick={() => open(active.lesson, active.step)}>
            ↺ Adımı yeniden kur
          </button>
        ) : (
          !finished && (
            <button
              type="button"
              className="primary-button"
              disabled={pending || !snapshot}
              onClick={() => onTrain(status === 'running' ? 'pause' : 'start')}
            >
              {status === 'running' ? 'Ⅱ Duraklat' : status === 'paused' ? '▶ Devam et' : '▶ Eğit'}
            </button>
          )
        )}
        {finished && matches && (
          <button
            type="button"
            className="primary-button"
            onClick={() => (last ? finish(lesson.id) : open(active.lesson, active.step + 1))}
          >
            {last ? '✓ Dersi bitir' : 'Sonraki adım →'}
          </button>
        )}
        {active.step > 0 && (
          <button type="button" onClick={() => open(active.lesson, active.step - 1)}>
            ← Önceki adım
          </button>
        )}
        {!finished && !last && (
          <button type="button" className="text-button" onClick={() => open(active.lesson, active.step + 1)}>
            Adımı atla
          </button>
        )}
      </div>
    </section>
  );
}
