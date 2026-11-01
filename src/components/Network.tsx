import { useMemo, useState } from 'react';
import type { Model, NeuronMaps, Selection, Trace } from '../types';
import { FEATURES, RAW_FEATURES } from '../ml/features';
import { HELP, Help } from './Help';
import { activationImage, probabilityImage } from './maps';
/** One data URL per neuron, indexed like trace layers: inputs, hidden layers, then the output probability map. */
export function useMapImages(view: { model: Model; neuronMaps: NeuronMaps; grid: number[]; resolution: number } | null): string[][] {
  return useMemo(() => view ? [
    ...view.neuronMaps.maps.map((layer, k) => layer.map(values => activationImage(values, view.neuronMaps.resolution, k === 0 ? 'input' : view.model.activation))),
    [probabilityImage(view.grid, view.resolution)],
  ] : [], [view?.neuronMaps, view?.grid, view?.resolution, view?.model.activation]);
}
export function Network({ model, trace, selection, onSelect, images }: { model: Model; trace: Trace; selection: Selection; onSelect: (s: Selection) => void; images: string[][] }) {
  const [showMaps, setShowMaps] = useState(true);
  const sizes = [model.layers[0].w[0].length, ...model.layers.map(l => l.b.length)], W = 560, H = 400;
  const features = model.features ?? RAW_FEATURES;
  const pos = (l: number, j: number): [number, number] => [84 + l / (sizes.length - 1) * (W - 150), 47 + (j + 0.5) / sizes[l] * (H - 83)];
  // Map tiles grow to fill the vertical space a layer leaves per neuron, within readable limits.
  const tile = (l: number) => Math.max(12, Math.min(38, (H - 83) / sizes[l] - 9));
  return <section className="panel network-panel">
    <div className="card-heading"><div><span className="eyebrow">MODELİN İÇİNDE</span><h2>Sinir ağı<Help label="Sinir ağı paneli" text={HELP.network}/></h2></div><span className="chip">{sizes.join(' : ')}</span></div>
    <div className="plot-legend"><span className="class-zero">━ Pozitif ağırlık</span><span className="class-one">━ Negatif ağırlık</span><span>Kalınlık = |w|</span>
      <label className="map-toggle"><input type="checkbox" checked={showMaps} onChange={e => setShowMaps(e.target.checked)}/>Nöron haritaları<Help label="Nöron haritaları" text={HELP.maps}/></label></div>
    <svg className="network-svg" viewBox={`0 0 ${W} ${H}`} role="group" aria-label="Gerçek ağırlık ve aktivasyonlarla sinir ağı. Nöronlar ve bağlantılar klavyeyle seçilebilir.">
      {sizes.map((_, l) => <text key={l} x={pos(l, 0)[0]} y="22" textAnchor="middle" className="layer-label">{l === 0 ? 'GİRİŞ' : l === sizes.length - 1 ? 'ÇIKIŞ' : `GİZLİ ${l}`}</text>)}
      {model.layers.flatMap((layer, l) => layer.w.flatMap((row, j) => row.map((w, i) => {
        const a = pos(l, i), b = pos(l + 1, j), chosen = selection.layer === l && selection.neuron === j && (selection.input === undefined || selection.input === i);
        const select = () => onSelect({ layer: l, neuron: j, input: i });
        return <g key={`${l}-${j}-${i}`}>
          <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke={w >= 0 ? '#72cbd5' : '#e1a66c'} strokeWidth={Math.min(4, 0.5 + Math.abs(w) * 0.75)} opacity={chosen ? 0.95 : 0.08 + Math.min(0.28, Math.abs(w) * 0.075)}/>
          <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="transparent" strokeWidth="9" role="button" tabIndex={chosen && selection.input === i ? 0 : -1} aria-label={`Katman ${l + 1}, nöron ${j + 1}, giriş ${i + 1}, ağırlık ${w.toFixed(4)}`} onClick={select} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(); } }} className="connection-target"/>
        </g>;
      })))}
      {sizes.flatMap((count, l) => Array.from({ length: count }, (_, j) => {
        const [x, y] = pos(l, j), a = trace.a[l][j], chosen = l === selection.layer + 1 && j === selection.neuron, s = tile(l);
        const select = () => { if (l > 0) onSelect({ layer: l - 1, neuron: j }); };
        const name = l === 0 ? FEATURES[features[j]].label : undefined;
        return <g key={`${l}-${j}`} role={l > 0 ? 'button' : undefined} tabIndex={l > 0 ? (chosen ? 0 : -1) : undefined} aria-label={l > 0 ? `Katman ${l}, nöron ${j + 1}, aktivasyon ${a.toFixed(4)}` : `Giriş ${name}`} onClick={select} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(); } }} className="neuron-target">
          {showMaps ? <>
            {chosen && <rect x={x - s / 2 - 4} y={y - s / 2 - 4} width={s + 8} height={s + 8} rx="4" fill="none" stroke="#d1e9b8" strokeWidth="1.5"/>}
            <image href={images[l]?.[j]} x={x - s / 2} y={y - s / 2} width={s} height={s} preserveAspectRatio="none"/>
            <rect x={x - s / 2} y={y - s / 2} width={s} height={s} rx="2" fill="none" stroke={chosen ? '#d1e9b8' : '#4a585e'} strokeWidth={chosen ? 1.5 : 0.8}/>
          </> : <>
            {chosen && <circle cx={x} cy={y} r="15" fill="none" stroke="#d1e9b8" strokeWidth="1"/>}
            <circle cx={x} cy={y} r={count > 10 ? 7.5 : 10} fill={a >= 0 ? '#284952' : '#513c2d'} stroke={chosen ? '#d1e9b8' : a >= 0 ? '#7acbd4' : '#e1a66c'} strokeWidth={chosen ? 2 : 1} fillOpacity={0.3 + Math.min(0.7, Math.abs(a))}/>
          </>}
          {l === 0 && <text x={x - (showMaps ? s / 2 + 6 : 23)} y={y + 4} textAnchor="end" className="node-label">{name}</text>}
          {l === sizes.length - 1 && <text x={x + (showMaps ? s / 2 + 6 : 18)} y={y + 4} className="node-label">{a.toFixed(2)}</text>}
        </g>;
      }))}
    </svg>
    <div className="network-footer"><span>Nöron veya bağlantı seç</span><span>{model.layers.reduce((n, l) => n + l.w.flat().length + l.b.length, 0)} öğrenilebilir parametre</span></div>
  </section>;
}
