import { useId, useState } from 'react';
import type { Config } from '../types';
import { DATASETS } from '../ml/config';
import { FEATURES, FEATURE_ORDER } from '../ml/features';
import { HELP, Help } from './Help';
export function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  help,
  disabled,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  help?: string;
  disabled?: boolean;
  onChange: (n: number) => void;
}) {
  const id = useId();
  // Decimal fields omit HTML min/max: the browser would use min as step base and mark valid values like 0.02 invalid. commit() clamps instead.
  const [draft, setDraft] = useState(String(value));
  // A new value from outside replaces whatever is being typed.
  const [shown, setShown] = useState(value);
  if (shown !== value) {
    setShown(value);
    setDraft(String(value));
  }
  const commit = () => {
    const n = Number(draft);
    if (!draft.trim() || !Number.isFinite(n)) {
      setDraft(String(value));
      return;
    }
    const next = Math.max(min, Math.min(max, step === 1 ? Math.round(n) : n));
    setDraft(String(next));
    if (next !== value) onChange(next);
  };
  return (
    // htmlFor is required: without it the label would attach to the help button, its first labelable child.
    <label className="field" htmlFor={id}>
      <span>
        {label}
        {help && <Help label={label} text={help} />}
      </span>
      <input
        id={id}
        type="number"
        min={step === 1 ? min : undefined}
        max={step === 1 ? max : undefined}
        step={step}
        disabled={disabled}
        value={draft}
        onChange={e => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={e => {
          if (e.key === 'Enter') {
            commit();
            e.currentTarget.blur();
          }
        }}
      />
    </label>
  );
}
export function Controls({ config: c, onChange }: { config: Config; onChange: (c: Config) => void }) {
  const patch = (p: Partial<Config>) => onChange({ ...c, ...p }),
    drawn = c.dataset === 'custom';
  return (
    <aside className="controls panel" aria-label="Deney kontrolleri">
      <div className="panel-title">
        <span className="eyebrow">DENEY DÜZENİ</span>
        <span className="tiny-dot" />
      </div>
      <section className="control-section">
        <h3>
          <span>01</span> Veri kümesi
        </h3>
        <label className="field" htmlFor="field-dataset">
          <span>
            Problem
            <Help label="Problem" text={HELP.dataset} />
          </span>
          <select
            id="field-dataset"
            value={c.dataset}
            onChange={e => patch({ dataset: e.target.value as Config['dataset'] })}
          >
            {Object.entries(DATASETS).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <div className="two-fields">
          <NumberField
            label="Örnek sayısı"
            help={HELP.samples}
            disabled={drawn}
            value={c.samples}
            min={40}
            max={2000}
            onChange={samples => patch({ samples })}
          />
          <NumberField
            label="Seed"
            help={HELP.seed}
            value={c.seed}
            min={0}
            max={4294967295}
            onChange={seed => patch({ seed })}
          />
        </div>
        <label className="range-field" htmlFor="field-noise">
          <span>
            <span>
              Gürültü
              <Help label="Gürültü" text={HELP.noise} />
            </span>
            <b>{c.noise.toFixed(3)}</b>
          </span>
          <input
            id="field-noise"
            aria-label="Veri gürültüsü"
            disabled={drawn}
            type="range"
            min="0"
            max="0.35"
            step="0.005"
            value={c.noise}
            onChange={e => patch({ noise: Number(e.target.value) })}
          />
          <div className="range-ends">
            <span>Temiz</span>
            <span>Gürültülü</span>
          </div>
        </label>
        {drawn && (
          <p className="quiet-note">
            Kendi çiziminde örnek sayısı ve gürültü kullanılmaz; noktalar tam koyduğun yerde durur.
          </p>
        )}
        <label className="check-field">
          <input type="checkbox" checked={c.split} onChange={e => patch({ split: e.target.checked })} />
          <span>
            %80 eğitim / %20 test ayır
            <Help label="Test bölmesi" text={HELP.split} />
          </span>
        </label>
      </section>
      <section className="control-section">
        <h3>
          <span>02</span> Ağ mimarisi
        </h3>
        <label className="field" htmlFor="field-layer-count">
          <span>
            Gizli katman sayısı
            <Help label="Gizli katman sayısı" text={HELP.layers} />
          </span>
          <select
            id="field-layer-count"
            value={c.hidden.length}
            onChange={e => {
              const n = Number(e.target.value);
              patch({ hidden: Array.from({ length: n }, (_, i) => c.hidden[i] ?? 8) });
            }}
          >
            <option>0</option>
            <option>1</option>
            <option>2</option>
            <option>3</option>
          </select>
        </label>
        <div className="layer-fields">
          {c.hidden.map((n, i) => (
            <NumberField
              key={i}
              label={`Katman ${i + 1} · nöron`}
              help={HELP.neurons}
              value={n}
              min={2}
              max={16}
              onChange={v => patch({ hidden: c.hidden.map((old, k) => (k === i ? v : old)) })}
            />
          ))}
        </div>
        <label className="field" htmlFor="field-activation">
          <span>
            Aktivasyon (activation)
            <Help label="Aktivasyon" text={HELP.activation} />
          </span>
          <select
            id="field-activation"
            value={c.activation}
            onChange={e => patch({ activation: e.target.value as Config['activation'] })}
          >
            <option value="tanh">tanh</option>
            <option value="relu">ReLU</option>
            <option value="sigmoid">Sigmoid</option>
          </select>
        </label>
        <div className="feature-field">
          <span>
            Girdi özellikleri
            <Help label="Girdi özellikleri" text={HELP.features} />
          </span>
          <div className="feature-chips" role="group" aria-label="Girdi özellikleri">
            {FEATURE_ORDER.map(f => {
              const on = c.features.includes(f);
              return (
                <button
                  key={f}
                  type="button"
                  aria-pressed={on}
                  className={on ? 'active' : ''}
                  disabled={on && c.features.length === 1}
                  title={on && c.features.length === 1 ? 'En az bir girdi gerekli' : undefined}
                  onClick={() =>
                    patch({ features: on ? c.features.filter(x => x !== f) : [...c.features, f] })
                  }
                >
                  {FEATURES[f].label}
                </button>
              );
            })}
          </div>
        </div>
        <p className="quiet-note">
          {c.features.length} giriş → {c.hidden.length ? `${c.hidden.join(' → ')} → ` : ''}1 sigmoid çıkış
        </p>
      </section>
      <section className="control-section">
        <h3>
          <span>03</span> Öğrenme
        </h3>
        <label className="field" htmlFor="field-optimizer">
          <span>
            Optimizer
            <Help label="Optimizer" text={HELP.optimizer} />
          </span>
          <select
            id="field-optimizer"
            value={c.optimizer}
            onChange={e => patch({ optimizer: e.target.value as Config['optimizer'] })}
          >
            <option value="adam">Adam</option>
            <option value="sgd">SGD</option>
          </select>
        </label>
        <NumberField
          label="Öğrenme hızı (learning rate)"
          help={HELP.learningRate}
          value={c.learningRate}
          min={0.0001}
          max={3}
          step={0.001}
          onChange={learningRate => patch({ learningRate })}
        />
        <input
          className="rate-range"
          aria-label="Öğrenme hızı kaydırıcısı"
          type="range"
          min="-4"
          max={Math.log10(3)}
          step="0.025"
          value={Math.log10(c.learningRate)}
          onChange={e => patch({ learningRate: Number((10 ** Number(e.target.value)).toPrecision(3)) })}
        />
        <div className="range-ends">
          <span>0.0001</span>
          <span>3.0</span>
        </div>
        <div className="two-fields">
          <NumberField
            label="Batch boyutu"
            help={HELP.batchSize}
            value={c.batchSize}
            min={1}
            max={256}
            onChange={batchSize => patch({ batchSize })}
          />
          <NumberField
            label="Hedef epoch"
            help={HELP.epochs}
            value={c.epochs}
            min={1}
            max={5000}
            onChange={epochs => patch({ epochs })}
          />
        </div>
        <NumberField
          label="L2 regularization"
          help={HELP.l2}
          value={c.l2}
          min={0}
          max={0.1}
          step={0.001}
          onChange={l2 => patch({ l2 })}
        />
      </section>
      <p className="settings-note">
        Bu kontroller değiştiğinde ağırlıklar ve geçmiş yeniden oluşturulur. Yalnızca gözlem hızı eğitimi
        korur.
      </p>
    </aside>
  );
}
