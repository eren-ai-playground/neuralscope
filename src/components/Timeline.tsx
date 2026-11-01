import { useEffect, useState } from 'react';
import { HELP, Help } from './Help';
/** Scrubs through stored epoch states. value null means the live model is shown. */
export function Timeline({ epochs, value, onChange, disabled }: { epochs: number[]; value: number | null; onChange: (epoch: number | null) => void; disabled: boolean }) {
  const [playing, setPlaying] = useState(false);
  const index = value === null ? epochs.length - 1 : Math.max(0, epochs.findIndex(e => e >= value));
  useEffect(() => { if (disabled) setPlaying(false); }, [disabled]);
  useEffect(() => {
    if (!playing) return;
    // Playback restarts from the beginning when it is already at the end.
    // Long runs skip frames so a full playback takes about 90 steps (~6 s).
    let i = index >= epochs.length - 1 ? 0 : index;
    const stride = Math.max(1, Math.ceil(epochs.length / 90));
    const timer = setInterval(() => {
      if (i >= epochs.length - 1) { setPlaying(false); return; }
      i = Math.min(epochs.length - 1, i + stride); onChange(epochs[i]);
    }, 70);
    onChange(epochs[i]);
    return () => clearInterval(timer);
    // Only (re)start when playback is toggled; index changes come from playback itself.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing]);
  if (epochs.length < 2) return null;
  return <section className="timeline panel" aria-label="Zaman çizelgesi">
    <div className="timeline-head"><span className="eyebrow">ZAMANDA GERİ SAR<Help label="Zaman çizelgesi" text={HELP.timeline}/></span>
      <strong>{value === null ? `Canlı · epoch ${epochs.at(-1)}` : `Epoch ${epochs[index]} / ${epochs.at(-1)}`}</strong></div>
    <div className="timeline-row">
      <button type="button" disabled={disabled} onClick={() => setPlaying(p => !p)} aria-label={playing ? 'Oynatmayı durdur' : 'Eğitimi baştan oynat'}>{playing ? 'Ⅱ' : '▶'}</button>
      <input type="range" min={0} max={epochs.length - 1} step={1} value={index} disabled={disabled} aria-label="Gösterilen epoch" aria-valuetext={`Epoch ${epochs[index]}`}
        onChange={e => { setPlaying(false); const i = Number(e.target.value); onChange(i === epochs.length - 1 ? null : epochs[i]); }}/>
      <button type="button" disabled={value === null} onClick={() => { setPlaying(false); onChange(null); }}>Canlıya dön</button>
    </div>
    {disabled && <p className="quiet-note">Eğitim çalışırken geri sarılamaz; duraklatınca kaydırıcı açılır.</p>}
  </section>;
}
