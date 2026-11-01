import type { EpochMetric } from '../types';
import type { PreviousRun } from '../hooks/useExperiment';
import { HELP, Help } from './Help';
const thin = (history: EpochMetric[]) =>
  history.length <= 650
    ? history
    : history.filter((_, i) => i === history.length - 1 || i % Math.ceil(history.length / 650) === 0);
export function Chart({
  history,
  previous,
  marker,
  kind,
}: {
  history: EpochMetric[];
  previous: EpochMetric[] | null;
  marker: number | null;
  kind: 'loss' | 'accuracy';
}) {
  const W = 520,
    H = 168,
    left = 45,
    right = 15,
    top = 16,
    bottom = 26;
  const indices = thin(history),
    ghost = previous ? thin(previous) : [];
  const pick = (m: EpochMetric) =>
    kind === 'loss' ? [m.trainLoss, m.testLoss] : [m.trainAccuracy, m.testAccuracy];
  const series = [indices.map(m => pick(m)[0]), indices.map(m => pick(m)[1])],
    ghostSeries = ghost.map(m => pick(m)[0]);
  const maxY =
    kind === 'loss'
      ? Math.max(0.1, ...[...series.flat(), ...ghostSeries].filter((v): v is number => v !== null)) * 1.08
      : 1;
  // The x axis covers both runs so the previous curve can be compared epoch for epoch.
  const maxX = Math.max(1, history.at(-1)?.epoch ?? 1, ghost.at(-1)?.epoch ?? 1);
  const X = (n: number) => left + (n / maxX) * (W - left - right),
    Y = (n: number) => top + (1 - n / maxY) * (H - top - bottom);
  const path = (values: (number | null)[], points: EpochMetric[]) =>
    values
      .map((v, i) =>
        v === null ? '' : `${i ? 'L' : 'M'}${X(points[i].epoch).toFixed(2)},${Y(v).toFixed(2)}`,
      )
      .join(' ');
  const last = history.at(-1);
  return (
    <div className="chart-block">
      <div className="chart-title">
        <h3>{kind === 'loss' ? 'Kayıp (BCE)' : 'Doğruluk'}</h3>
        <span>
          {last
            ? kind === 'loss'
              ? last.trainLoss.toFixed(4)
              : `${(last.trainAccuracy * 100).toFixed(1)}%`
            : '—'}
        </span>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${kind === 'loss' ? 'Kayıp' : 'Doğruluk'} grafiği, tamamlanmış epoch ölçümleri. Eğitim ${last ? (kind === 'loss' ? last.trainLoss.toFixed(4) : (last.trainAccuracy * 100).toFixed(1) + '%') : 'henüz ölçülmedi'}${previous ? '. Önceki deney gri çizgiyle gösteriliyor' : ''}`}
      >
        {[0, 0.5, 1].map(t => (
          <g key={t}>
            <line
              x1={left}
              y1={Y(t * maxY)}
              x2={W - right}
              y2={Y(t * maxY)}
              stroke="#293137"
              strokeDasharray="3 5"
            />
            <text x={left - 9} y={Y(t * maxY) + 4} textAnchor="end">
              {kind === 'loss' ? (t * maxY).toFixed(2) : `${t * 100}%`}
            </text>
          </g>
        ))}
        {ghost.length > 0 && (
          <path
            d={path(ghostSeries, ghost)}
            stroke="#8b9aa1"
            strokeWidth="2"
            strokeOpacity="0.55"
            fill="none"
          />
        )}
        <path d={path(series[0], indices)} stroke="#a9d98b" strokeWidth="2" fill="none" />
        <path
          d={path(series[1], indices)}
          stroke="#e1a66c"
          strokeWidth="1.8"
          strokeDasharray="5 4"
          fill="none"
        />
        {indices.length === 1 &&
          series.map(
            (s, i) =>
              s[0] !== null && (
                <circle key={i} cx={X(0)} cy={Y(s[0])} r="3" fill={i ? '#e1a66c' : '#a9d98b'} />
              ),
          )}
        {marker !== null && (
          <line
            className="chart-marker"
            x1={X(marker)}
            y1={top}
            x2={X(marker)}
            y2={H - bottom}
            stroke="#f2f5f0"
            strokeWidth="1.2"
            strokeDasharray="2 3"
          />
        )}
        <text x={left} y={H - 6}>
          0
        </text>
        <text x={W - right} y={H - 6} textAnchor="end">
          {maxX} epoch
        </text>
      </svg>
    </div>
  );
}
export function Charts({
  history,
  hasTest,
  previous,
  onClearPrevious,
  marker,
}: {
  history: EpochMetric[];
  hasTest: boolean;
  previous: PreviousRun | null;
  onClearPrevious: () => void;
  marker: number | null;
}) {
  return (
    <section className="panel charts-panel">
      <div className="card-heading">
        <div>
          <span className="eyebrow">ÖĞRENME GEÇMİŞİ</span>
          <h2>
            Her epoch bir iz bırakır
            <Help label="Öğrenme grafikleri" text={HELP.charts} />
          </h2>
        </div>
        <span className="chart-legend">
          <span>━ Eğitim</span>
          {hasTest && <span className="legend-test">┄ Test</span>}
          {previous && <span className="legend-previous">━ Önceki</span>}
        </span>
      </div>
      {previous && (
        <div className="compare-bar">
          <span>
            <b>Önceki deneyle karşılaştırma</b>
            <Help label="Karşılaştırma" text={HELP.compare} /> {previous.change}
          </span>
          <button type="button" onClick={onClearPrevious}>
            Kaldır
          </button>
        </div>
      )}
      <div className="charts-grid">
        <Chart history={history} previous={previous?.history ?? null} marker={marker} kind="loss" />
        <Chart history={history} previous={previous?.history ?? null} marker={marker} kind="accuracy" />
      </div>
      <p className="quiet-note">
        Tam kümelerden alınan epoch sonu ölçümleri. BCE grafiği L2 cezasını içermez.
        {previous && ' Gri çizgi önceki deneyin eğitim eğrisidir.'}
      </p>
    </section>
  );
}
