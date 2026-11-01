import { useEffect, useRef } from 'react';
import type { Point, Snapshot } from '../types';
import { EXTENT } from '../ml/datasets';
import { HELP, Help } from './Help';
import { PROBABILITY_COLORS } from './maps';
export type DrawTool = 'zero' | 'one' | 'erase' | 'probe';
const TOOLS: [DrawTool, string][] = [['zero', '● Sınıf 0'], ['one', '▲ Sınıf 1'], ['erase', '⌫ Sil'], ['probe', '⌖ İncele']];
/** draw is set only for the custom dataset; clicks then edit points unless the probe tool is active. */
export function DecisionBoundary({ snapshot, train, test, probe, onProbe, draw, replayEpoch }: { snapshot: Snapshot | null; train: Point[]; test: Point[]; probe: Point; onProbe: (p: Point) => void; replayEpoch: number | null;
  draw: { tool: DrawTool; onTool: (t: DrawTool) => void; onEdit: (x: [number, number], tool: DrawTool) => void; onClear: () => void } | null }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const el = canvas.current; if (!el) return; const ctx = el.getContext('2d'); if (!ctx) return;
    const S = 520, padding = 35, W = S - padding * 2;
    el.width = S * 2; el.height = S * 2; ctx.setTransform(2, 0, 0, 2, 0, 0);
    ctx.fillStyle = '#111b21'; ctx.fillRect(0, 0, S, S);
    const X = (x: number) => padding + (x + EXTENT) / (EXTENT * 2) * W;
    const Y = (y: number) => padding + (EXTENT - y) / (EXTENT * 2) * W;
    if (snapshot) {
      const n = snapshot.resolution, heat = document.createElement('canvas'); heat.width = n; heat.height = n;
      const h = heat.getContext('2d')!, img = h.createImageData(n, n);
      snapshot.grid.forEach((p, i) => {
        const [color0, color1] = PROBABILITY_COLORS;
        for (let k = 0; k < 3; k++) img.data[i * 4 + k] = color0[k] * (1 - p) + color1[k] * p;
        img.data[i * 4 + 3] = 255;
      });
      h.putImageData(img, 0, 0); ctx.imageSmoothingEnabled = true; ctx.drawImage(heat, padding, padding, W, W);
      // Marching squares on actual model probabilities, with interpolated 0.5 crossings.
      ctx.strokeStyle = '#d7e5d7'; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]); ctx.beginPath();
      for (let y = 0; y < n - 1; y++) for (let x = 0; x < n - 1; x++) {
        const corners = [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]];
        const crossings: number[][] = [];
        for (let e = 0; e < 4; e++) {
          const a = corners[e], b = corners[(e + 1) % 4], va = snapshot.grid[a[1] * n + a[0]], vb = snapshot.grid[b[1] * n + b[0]];
          if ((va >= 0.5) !== (vb >= 0.5)) {
            const t = (0.5 - va) / (vb - va);
            crossings.push([padding + (a[0] + (b[0] - a[0]) * t + 0.5) / n * W, padding + (a[1] + (b[1] - a[1]) * t + 0.5) / n * W]);
          }
        }
        for (let e = 0; e + 1 < crossings.length; e += 2) { ctx.moveTo(...crossings[e] as [number, number]); ctx.lineTo(...crossings[e + 1] as [number, number]); }
      }
      ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.strokeStyle = '#c1d4d41a'; ctx.lineWidth = 1; ctx.font = '11px Consolas, monospace';
    for (const v of [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]) {
      ctx.beginPath(); ctx.moveTo(X(v), padding); ctx.lineTo(X(v), S - padding); ctx.moveTo(padding, Y(v)); ctx.lineTo(S - padding, Y(v)); ctx.stroke();
      ctx.fillStyle = '#9cabb2'; ctx.textAlign = 'center'; ctx.fillText(String(v), X(v), S - 14); ctx.textAlign = 'right'; ctx.fillText(String(v), padding - 8, Y(v) + 3);
    }
    ctx.save(); ctx.beginPath(); ctx.rect(padding, padding, W, W); ctx.clip();
    const dot = (p: Point, heldOut: boolean) => {
      const x = X(p.x[0]), y = Y(p.x[1]); ctx.beginPath();
      if (p.y) { ctx.moveTo(x, y - 4); ctx.lineTo(x + 4, y + 3); ctx.lineTo(x - 4, y + 3); ctx.closePath(); }
      else ctx.arc(x, y, 3.4, 0, 2 * Math.PI);
      ctx.fillStyle = heldOut ? '#111a20' : p.y ? '#edb36b' : '#76d5e2'; ctx.fill();
      ctx.strokeStyle = heldOut ? (p.y ? '#ffcb88' : '#98eeff') : '#111a20'; ctx.lineWidth = heldOut ? 1.5 : 0.8; ctx.stroke();
    };
    train.forEach(p => dot(p, false)); test.forEach(p => dot(p, true));
    const px = X(probe.x[0]), py = Y(probe.x[1]); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.arc(px, py, 9, 0, Math.PI * 2); ctx.moveTo(px - 14, py); ctx.lineTo(px + 14, py); ctx.moveTo(px, py - 14); ctx.lineTo(px, py + 14); ctx.stroke(); ctx.restore();
    ctx.fillStyle = '#a8b6bd'; ctx.textAlign = 'center'; ctx.fillText('x₁', S - 9, S - 14); ctx.fillText('x₂', 18, 15);
  }, [snapshot, train, test, probe]);
  const outside = [...train, ...test].filter(p => p.x.some(v => Math.abs(v) > EXTENT)).length;
  return <section className="panel decision-panel">
    <div className="card-heading"><div><span className="eyebrow">GİRDİ UZAYI</span><h2>Karar sınırı<Help label="Karar sınırı" text={HELP.decision}/></h2></div><span className="chip">{replayEpoch !== null ? `geri sarıldı · epoch ${replayEpoch}` : '2D · canlı hesap'}</span></div>
    <div className="plot-legend"><span className="class-zero">● Sınıf 0</span><span className="class-one">▲ Sınıf 1</span><span>İçi boş = test</span><span>┄ p = 0.5</span></div>
    <canvas ref={canvas} className="decision-canvas" tabIndex={0} role="application" aria-label="Karar sınırı. Tıklayarak ya da ok tuşlarıyla inceleme girdisini seçin."
      onClick={e => { const r = e.currentTarget.getBoundingClientRect(); const x = (e.clientX - r.left) / r.width * 520, y = (e.clientY - r.top) / r.height * 520; if (x < 35 || x > 485 || y < 35 || y > 485) return; const point: [number, number] = [(x - 35) / 450 * 3 - 1.5, 1.5 - (y - 35) / 450 * 3]; if (draw && draw.tool !== 'probe') draw.onEdit(point, draw.tool); else onProbe({ x: point, y: probe.y, id: -1 }); }}
      onKeyDown={e => { if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) return; e.preventDefault(); const x: [number, number] = [...probe.x]; const i = e.key === 'ArrowLeft' || e.key === 'ArrowRight' ? 0 : 1; x[i] = Math.max(-1.5, Math.min(1.5, x[i] + (e.key === 'ArrowLeft' || e.key === 'ArrowDown' ? -0.05 : 0.05))); onProbe({ x, y: probe.y, id: -1 }); }}/>
    {draw && <div className="draw-tools" role="group" aria-label="Çizim aracı"><span>Çizim<Help label="Kendi verini çiz" text={HELP.custom}/></span>{TOOLS.map(([tool, label]) => <button key={tool} type="button" aria-pressed={draw.tool === tool} className={draw.tool === tool ? `active tool-${tool}` : `tool-${tool}`} onClick={() => draw.onTool(tool)}>{label}</button>)}<button type="button" className="draw-clear" disabled={!train.length && !test.length} onClick={draw.onClear}>Temizle</button><small>{train.length + test.length} nokta</small></div>}
    <div className="probability-scale"><span>0</span><div/><span>1</span><small>P(sınıf 1)</small></div>
    <p className="plot-help">{draw && draw.tool !== 'probe' ? (draw.tool === 'erase' ? 'Silmek istediğin noktaya tıkla.' : `Tıkladığın yere sınıf ${draw.tool === 'one' ? 1 : 0} noktası eklenir.`) : 'Bir koordinata tıkla; ağın o girdiyi nasıl işlediğini incele.'}{outside > 0 && ` ${outside} nokta çizim alanının dışında; eğitim ve ölçüme dahil.`}</p>
  </section>;
}
