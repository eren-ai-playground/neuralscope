import type { Config, Point, Selection, Snapshot, Trace } from '../types';
import { backward } from '../ml/model';
import { NumberField } from './Controls';
import { FEATURES, RAW_FEATURES } from '../ml/features';
import { HELP, Help } from './Help';
const fmt = (n: number | undefined) =>
  n === undefined ? '—' : Math.abs(n) > 0 && Math.abs(n) < 0.0001 ? n.toExponential(2) : n.toFixed(4);
// Display-only rounding: the probe keeps its exact coordinate unless the user edits the field.
const round = (n: number) => Math.round(n * 1000) / 1000;
export function Inspector({
  snapshot,
  trace,
  selection: s,
  probe,
  onProbe,
  onSelect,
  config,
  mapImage,
  replayEpoch,
}: {
  snapshot: Snapshot;
  trace: Trace;
  selection: Selection;
  probe: Point;
  onProbe: (p: Point | null) => void;
  onSelect: (s: Selection) => void;
  config: Config;
  mapImage?: string;
  replayEpoch: number | null;
}) {
  const l = snapshot.model.layers[s.layer],
    j = s.neuron,
    gradients = backward(snapshot.model, probe, trace);
  const activation = s.layer === snapshot.model.layers.length - 1 ? 'sigmoid' : config.activation;
  const incoming = trace.a[s.layer],
    chosenInput = s.input ?? 0,
    features = snapshot.model.features ?? RAW_FEATURES;
  const output = s.layer === snapshot.model.layers.length - 1;
  const gradient = gradients.layers[s.layer].w[j][chosenInput],
    update = snapshot.update?.[s.layer].w[j][chosenInput];
  return (
    <aside className="panel inspector-panel" aria-label="Nöron inceleyici">
      <div className="card-heading">
        <div>
          <span className="eyebrow">HESABI TAKİP ET</span>
          <h2>Nöron inceleyici</h2>
        </div>
        <span className="inspector-icon">◎</span>
      </div>
      <div className="selected-neuron">
        <span className="neuron-badge">N{j + 1}</span>
        <div>
          <strong>
            Katman {s.layer + 1} · Nöron {j + 1}
          </strong>
          <span>
            {activation} aktivasyonu{s.input !== undefined ? ` · bağlantı ${s.input + 1}` : ''}
          </span>
        </div>
      </div>
      {mapImage && (
        <div className="inspector-section neuron-map">
          <h3>
            {output ? 'Çıkışın haritası' : 'Bu nöronun haritası'}
            <Help label="Nöron haritası" text={output ? HELP.decision : HELP.inspectorMap} />
          </h3>
          <img src={mapImage} alt={`Katman ${s.layer + 1}, nöron ${j + 1} için girdi uzayı haritası`} />
          <p className="quiet-note">
            Yatay x₁, dikey x₂; aralık −1.5 ile 1.5.
            {replayEpoch !== null ? ` Epoch ${replayEpoch} anındaki hali.` : ''}
          </p>
        </div>
      )}
      <div className="inspector-section inspector-selects">
        <div className="two-fields">
          <label className="field">
            <span>İncelenen katman</span>
            <select value={s.layer} onChange={e => onSelect({ layer: Number(e.target.value), neuron: 0 })}>
              {snapshot.model.layers.map((_, i) => (
                <option key={i} value={i}>
                  {i === snapshot.model.layers.length - 1 ? 'Çıkış' : `Gizli ${i + 1}`}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>İncelenen nöron</span>
            <select value={j} onChange={e => onSelect({ layer: s.layer, neuron: Number(e.target.value) })}>
              {l.b.map((_, i) => (
                <option key={i} value={i}>
                  Nöron {i + 1}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>
      <div className="inspector-section">
        <h3>İncelenen girdi</h3>
        <p className="quiet-note">
          {probe.id < 0
            ? 'Seçtiğin koordinat. Türev için aşağıdaki hedef sınıf varsayılır.'
            : `${snapshot.batches ? 'Son batch' : 'Başlangıç'} örneği #${probe.id} · gerçek sınıf ${probe.y}`}
        </p>
        <div className="two-fields">
          <NumberField
            label="x₁"
            value={round(probe.x[0])}
            min={-1.5}
            max={1.5}
            step={0.01}
            onChange={v => onProbe({ ...probe, id: -1, x: [v, probe.x[1]] })}
          />
          <NumberField
            label="x₂"
            value={round(probe.x[1])}
            min={-1.5}
            max={1.5}
            step={0.01}
            onChange={v => onProbe({ ...probe, id: -1, x: [probe.x[0], v] })}
          />
        </div>
        {probe.id < 0 && (
          <div className="target-controls">
            <span>Varsayılan hedef y</span>
            {[0, 1].map(y => (
              <button
                key={y}
                aria-pressed={probe.y === y}
                className={probe.y === y ? 'active' : ''}
                onClick={() => onProbe({ ...probe, y })}
              >
                {y}
              </button>
            ))}
          </div>
        )}
        <button className="text-button" disabled={probe.id >= 0} onClick={() => onProbe(null)}>
          Son eğitim örneğine dön ↗
        </button>
      </div>
      <div className="inspector-section">
        <h3>
          İleri geçiş (forward pass)
          <Help label="İleri geçiş" text={HELP.inspectorForward} />
        </h3>
        <p className="formula">z = Σ(wᵢ · aᵢ) + b</p>
        <div className="weight-table">
          <table>
            <thead>
              <tr>
                <th scope="col">Girdi aᵢ</th>
                <th scope="col">wᵢ</th>
                <th scope="col">wᵢ · aᵢ</th>
              </tr>
            </thead>
            <tbody>
              {incoming.map((a, i) => (
                <tr key={i} className={i === s.input ? 'selected-row' : ''}>
                  <td>
                    <button
                      aria-label={`Giriş ${i + 1} bağlantısını incele`}
                      onClick={() => onSelect({ ...s, input: i })}
                    >
                      {s.layer === 0 ? `${FEATURES[features[i]].label} = ` : ''}
                      {fmt(a)}
                    </button>
                  </td>
                  <td>{fmt(l.w[j][i])}</td>
                  <td>{fmt(a * l.w[j][i])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <dl className="values">
          <div>
            <dt>Bias b</dt>
            <dd>{fmt(l.b[j])}</dd>
          </div>
          <div>
            <dt>Toplam z</dt>
            <dd>{fmt(trace.z[s.layer][j])}</dd>
          </div>
          <div className="value-highlight">
            <dt>a = {activation}(z)</dt>
            <dd>{fmt(trace.a[s.layer + 1][j])}</dd>
          </div>
        </dl>
      </div>
      <div className="inspector-section">
        <h3>
          Geri yayılım (backpropagation)
          <Help label="Geri yayılım" text={HELP.inspectorBackward} />
        </h3>
        <p className="quiet-note">
          Aynı girdinin mevcut ağırlıklarla hesaplanan tek örnek BCE türevleri. L2 dahil değil.
        </p>
        <dl className="values">
          <div>
            <dt>∂L/∂z = ∂L/∂b</dt>
            <dd>{fmt(gradients.dz[s.layer][j])}</dd>
          </div>
          <div>
            <dt>∂L/∂w{chosenInput + 1}</dt>
            <dd>{fmt(gradient)}</dd>
          </div>
          <div>
            <dt>SGD yönü −∂L/∂w</dt>
            <dd>{gradient === 0 ? 'Sıfır' : gradient > 0 ? '↓ Azalt' : '↑ Artır'}</dd>
          </div>
        </dl>
      </div>
      <div className="inspector-section">
        <h3>
          Son gerçek batch güncellemesi
          <Help label="Batch güncellemesi" text={HELP.inspectorUpdate} />
        </h3>
        <p className="quiet-note">
          Güncelleme öncesi batch ortalaması ve L2; ardından {config.optimizer.toUpperCase()} adımı.
        </p>
        <dl className="values">
          <div>
            <dt>Batch ∂L/∂w{chosenInput + 1}</dt>
            <dd>{fmt(snapshot.batchGradient?.[s.layer].w[j][chosenInput])}</dd>
          </div>
          <div>
            <dt>Gerçek Δw{chosenInput + 1}</dt>
            <dd>{fmt(update)}</dd>
          </div>
          <div>
            <dt>Gerçek Δb</dt>
            <dd>{fmt(snapshot.update?.[s.layer].b[j])}</dd>
          </div>
        </dl>
      </div>
      <div className="prediction">
        <span>Ağın çıkışı · P(sınıf 1)</span>
        <strong>
          {(trace.a.at(-1)![0] * 100).toFixed(2)}
          <small>%</small>
        </strong>
        <span>Tahmin: sınıf {Number(trace.a.at(-1)![0] >= 0.5)} · eşik 0.5</span>
      </div>
    </aside>
  );
}
