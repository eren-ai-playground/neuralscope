import { useEffect, useMemo, useRef, useState } from 'react';
import type { Config, Layer, Point, Selection } from './types';
import { DATASETS, DEFAULT, MAX_POINTS, PRESETS, experimentKey, validateConfig } from './ml/config';
import { generateData, splitData } from './ml/datasets';
import { forward } from './ml/model';
import { useExperiment } from './hooks/useExperiment';
import { Controls } from './components/Controls';
import { HELP, Help } from './components/Help';
import { DecisionBoundary, type DrawTool } from './components/DecisionBoundary';
import { Network, useMapImages } from './components/Network';
import { Timeline } from './components/Timeline';
import { Inspector } from './components/Inspector';
import { Charts } from './components/Charts';
import { download, exportConfig, exportModel, historyCSV, importExperiment } from './utils/files';
const STORE = 'neuralscope.config.v1';
const round = (n: number) => Math.round(n * 1000) / 1000;
function initialConfig(): Config { try { const text = localStorage.getItem(STORE); return text ? validateConfig(JSON.parse(text)) : { ...DEFAULT, hidden: [...DEFAULT.hidden] }; } catch { return { ...DEFAULT, hidden: [...DEFAULT.hidden] }; } }
export default function App() {
  const [config, setConfig] = useState<Config>(initialConfig), [reset, setReset] = useState(0), [notice, setNotice] = useState('');
  const [probe, setProbe] = useState<Point | null>(null), [selection, setSelection] = useState<Selection>({ layer: 0, neuron: 0 });
  const [fileError, setFileError] = useState(''); const fileInput = useRef<HTMLInputElement>(null);
  // weights: starting weights loaded from a model file; kept across reset, dropped when the experiment changes.
  const [weights, setWeights] = useState<Layer[] | null>(null), [replayEpoch, setReplayEpoch] = useState<number | null>(null), [tool, setTool] = useState<DrawTool>('zero');
  useEffect(() => { setProbe(null); setSelection({ layer: 0, neuron: 0 }); }, [reset]);
  const { snapshot, pending, error, problem, command, previous, clearPrevious, frame, showFrame } = useExperiment(config, reset, weights);
  const dataKey = JSON.stringify([config.dataset, config.samples, config.noise, config.seed, config.split, config.dataset === 'custom' ? config.customPoints : null]);
  const data = useMemo(() => splitData(generateData(config), config), [dataKey]);
  useEffect(() => { try { localStorage.setItem(STORE, JSON.stringify(config)); } catch { /* Browser privacy/storage policies may disable persistence. Training still works. */ } }, [config]);
  /** message overrides the default notice for a new experiment; an empty string shows none. */
  const change = (next: Config, message?: string) => {
    // Switching to drawing starts from the points currently on screen, so the user edits a familiar shape.
    if (next.dataset === 'custom' && config.dataset !== 'custom' && !next.customPoints.length) {
      next = { ...next, customPoints: [...data.train, ...data.test].sort((a, b) => a.id - b.id).map(p => ({ x: [round(p.x[0]), round(p.x[1])], y: p.y })) };
      message ??= 'Mevcut noktalar kendi çizimine kopyalandı. Karar sınırına tıklayarak nokta ekle ya da “Temizle” ile boş başla.'; setTool('zero');
    }
    setConfig(validateConfig(next));
    if (experimentKey(config) !== experimentKey(next)) { setProbe(null); setSelection({ layer: 0, neuron: 0 }); setWeights(null); setReplayEpoch(null); setNotice(message ?? 'Yeni deney hazır. Ağırlıklar seed ile yeniden üretildi; geçmiş temizlendi.'); }
  };
  const editPoints = (x: [number, number], action: DrawTool) => {
    const points = config.customPoints, distance = (p: { x: number[] }) => Math.hypot(p.x[0] - x[0], p.x[1] - x[1]);
    if (action === 'erase') {
      const nearest = points.reduce((best, p, i) => best < 0 || distance(p) < distance(points[best]) ? i : best, -1);
      if (nearest >= 0 && distance(points[nearest]) < 0.12) change({ ...config, customPoints: points.filter((_, i) => i !== nearest) }, '');
    } else if (points.length >= MAX_POINTS) setNotice(`En fazla ${MAX_POINTS} nokta çizilebilir.`);
    else change({ ...config, customPoints: [...points, { x: [round(x[0]), round(x[1])], y: action === 'one' ? 1 : 0 }] }, '');
  };
  // Training or a new run always returns to the live view.
  useEffect(() => { if (pending || snapshot?.status === 'running') { setReplayEpoch(null); showFrame(null); } }, [pending, snapshot?.status]);
  const timeline = (epoch: number | null) => { setReplayEpoch(epoch); showFrame(epoch); };
  const view = snapshot && frame && replayEpoch !== null ? { ...snapshot, model: frame.model, grid: frame.grid, resolution: frame.resolution, neuronMaps: frame.neuronMaps, batchGradient: null, update: null } : snapshot;
  const shownEpoch = view !== snapshot && frame ? frame.epoch : null, images = useMapImages(view);
  const activeProbe = probe ?? (pending ? undefined : snapshot?.sample) ?? data.train[0] ?? { x: [0, 0], y: 0, id: -1 };
  const trace = view ? forward(view.model, activeProbe.x) : null;
  const safeSelection = view && selection.layer < view.model.layers.length && selection.neuron < view.model.layers[selection.layer].b.length ? selection : { layer: 0, neuron: 0 };
  const last = snapshot?.history.at(-1), initial = snapshot?.history[0];
  const status = error ? 'error' : snapshot?.status ?? 'ready';
  const running = status === 'running', done = status === 'done';
  const statuses = { ready: 'Başlamaya hazır', running: 'Eğitim çalışıyor', paused: 'Duraklatıldı', done: 'Eğitim tamamlandı', error: 'Eğitim hatası' };
  const recent = snapshot?.history.slice(-8) ?? [];
  const rising = recent.length >= 6 && recent.filter((m, i) => i > 0 && m.trainLoss > recent[i - 1].trainLoss + 0.002).length >= 3;
  const gap = last?.testAccuracy !== null && last?.testAccuracy !== undefined ? last.trainAccuracy - last.testAccuracy : null;
  const tooSmall = ['spiral', 'circles', 'xor'].includes(config.dataset) && config.features.length === 2 && config.hidden.reduce((s, n) => s + n, 0) <= 4;
  const resetModel = () => { setReset(n => n + 1); setProbe(null); setSelection({ layer: 0, neuron: 0 }); setNotice(weights ? 'Deney sıfırlandı; yüklenen model ağırlıklarına dönüldü.' : 'Deney sıfırlandı. Aynı seed aynı başlangıç ağırlıklarını üretir.'); };
  const waiting = problem ?? 'Model hazırlanıyor…';
  return <div className="app-shell">
    <header className="topbar"><a className="brand" href="#main" aria-label="NeuralScope laboratuvarına git"><svg viewBox="0 0 36 36" aria-hidden="true"><path d="M8 10L18 5L28 12L25 27L10 29L8 10M8 10L25 27M18 5L10 29M28 12L10 29" fill="none" stroke="currentColor" strokeWidth="1.4"/>{[[8,10],[18,5],[28,12],[25,27],[10,29]].map(([x,y],i)=><circle key={i} cx={x} cy={y} r="2.4" fill="currentColor"/>)}</svg><span>Neural<span className="brand-light">Scope</span></span></a><div className="topbar-divider"/><span className="topbar-label">İNTERAKTİF SİNİR AĞI LABORATUVARI</span><span className="local-tag"><span/> Tarayıcıda · CPU</span></header>
    <main id="main"><div className="intro"><div><span className="eyebrow accent">DENEY YAP. GÖZLEMLE. ANLA.</span><h1>Öğrenmeyi gözlerinle gör.</h1><p>Bir sinir ağının kararları, ağırlıkları ve her küçük adımı.</p></div><div className={`status-pill ${status}`} role="status"><span/>{statuses[status]}</div></div>
      <nav className="presets" aria-label="Hızlı başlangıç deneyleri"><span className="presets-label">HIZLI BAŞLANGIÇ</span>{PRESETS.map((p, i) => <button key={p.name} title={p.description} aria-pressed={experimentKey(config) === experimentKey(p.config)} className={experimentKey(config) === experimentKey(p.config) ? 'preset active' : 'preset'} onClick={() => { change({ ...p.config, hidden: [...p.config.hidden] }); setReset(n => n + 1); }}><span className="preset-symbol">{['⊕','◒','◎','〰'][i]}</span>{p.name}<span className="preset-arrow">↗</span></button>)}</nav>
      <div className="training-toolbar"><div className="training-actions"><button className="primary-button" onClick={() => command(running ? 'pause' : 'start')} disabled={!snapshot || pending || done || !!error}>{running ? 'Ⅱ Duraklat' : status === 'paused' ? '▶ Devam et' : '▶ Başlat'}</button><button onClick={() => command('step')} disabled={!snapshot || pending || running || done || !!error}>▹ Tek batch</button><button onClick={resetModel}>↺ Sıfırla</button></div><div className="speed-control"><span>Gözlem hızı<Help label="Gözlem hızı" text={HELP.speed}/></span><div role="group" aria-label="Gözlem hızı">{(['slow','normal','fast'] as const).map((s, i) => <button key={s} aria-pressed={config.speed === s} className={config.speed === s ? 'active' : ''} onClick={() => change({ ...config, speed: s })}>{['Yavaş','Normal','Hızlı'][i]}</button>)}</div></div><span className="batch-count">{snapshot?.batches ?? 0} gerçek batch</span></div>
      {(error || fileError) && <div className="error-banner" role="alert">{error || fileError}{fileError && <button onClick={() => setFileError('')}>Kapat</button>}</div>}
      {notice && <div className="notice" role="status">{notice}<button aria-label="Bildirimi kapat" onClick={() => setNotice('')}>×</button></div>}
      <div className={pending ? 'metric-strip pending' : 'metric-strip'} aria-busy={pending} aria-label="Gerçek eğitim metrikleri"><div><span>Epoch<Help label="Epoch" text={HELP.metricEpoch}/></span><strong>{snapshot?.epoch ?? 0}<small> / {config.epochs}</small></strong><div className="epoch-progress"><i style={{ width: `${(snapshot?.epochProgress ?? 0) * 100}%` }}/></div></div><div><span>Eğitim kaybı <em>BCE</em><Help label="Eğitim kaybı" text={HELP.metricLoss}/></span><strong>{last?.trainLoss.toFixed(4) ?? '—'}</strong><small>Son tam epoch: {last?.epoch ?? '—'}</small></div><div><span>Eğitim doğruluğu<Help label="Eğitim doğruluğu" text={HELP.metricAccuracy}/></span><strong className="metric-green">{last ? (last.trainAccuracy * 100).toFixed(1) : '—'}<small>%</small></strong><small>{data.train.length} eğitim örneği</small></div><div><span>Test doğruluğu<Help label="Test doğruluğu" text={HELP.metricTest}/></span><strong>{last?.testAccuracy !== null && last?.testAccuracy !== undefined ? (last.testAccuracy * 100).toFixed(1) : '—'}<small>{config.split ? '%' : ''}</small></strong><small>{config.split ? `${data.test.length} ayrılmış örnek` : 'Test bölmesi kapalı'}</small></div><div><span>Aktif süre<Help label="Aktif süre" text={HELP.metricTime}/></span><strong>{((snapshot?.elapsedMs ?? 0) / 1000).toFixed(1)}<small> sn</small></strong><small>{config.optimizer.toUpperCase()} · η {config.learningRate}</small></div></div>
      <div className={pending ? 'workspace pending' : 'workspace'} aria-busy={pending}><Controls config={config} onChange={change}/><div className="workspace-center"><div className="visual-pair"><DecisionBoundary snapshot={view} train={data.train} test={data.test} probe={activeProbe} onProbe={setProbe} replayEpoch={shownEpoch}
          draw={config.dataset === 'custom' ? { tool, onTool: setTool, onEdit: editPoints, onClear: () => change({ ...config, customPoints: [] }, 'Tüm noktalar silindi. Karar sınırına tıklayarak yeni noktalar ekle.') } : null}/>
        {view && trace ? <Network model={view.model} trace={trace} selection={safeSelection} onSelect={setSelection} images={images}/> : <section className="panel loading-panel">{waiting}</section>}</div>
        {snapshot && <Timeline epochs={snapshot.frameEpochs} value={replayEpoch} onChange={timeline} disabled={running || pending}/>}
        <Charts history={snapshot?.history ?? []} hasTest={config.split} previous={previous} onClearPrevious={clearPrevious} marker={shownEpoch}/>
        <section className="learning-panel"><div className="card-heading"><div><span className="eyebrow">SAYILARDAN SEZGİYE</span><h2>Bu deney bize ne söylüyor?</h2></div><span className="chip">{DATASETS[config.dataset]}</span></div><div className="learning-grid">
          <article><span>01 / ZİNCİR KURALI</span><h3>Geri yayılım</h3><p>Çıkışta ∂L/∂z = p − y. Bu hata, ağırlıklar ve aktivasyon türevleriyle geriye taşınır. Seçili nöronun gerçek türevini inceleyicide takip et.</p></article>
          <article><span>02 / KÜÇÜK ADIMLAR</span><h3>Gradyan inişi</h3><p>{initial && last ? `BCE ${initial.trainLoss.toFixed(4)} değerinden ${last.trainLoss.toFixed(4)} değerine geldi. ` : ''}SGD, w ← w − η∂L/∂w uygular. Adam geçmiş gradyanların momentlerini de kullanır.</p></article>
          <article><span>03 / ADIM BOYU</span><h3>Öğrenme hızı</h3><p>{rising ? 'Son 8 epoch içinde loss en az 3 kez yükseldi. Daha küçük bir hızla aynı seed üzerinde yeni deney dene.' : config.learningRate > 0.3 ? `η = ${config.learningRate}: büyük adımlar dalgalanma oluşturabilir. Grafikte gerçekten oluşup oluşmadığını izle.` : `η = ${config.learningRate}. Hızı değiştirip aynı seed ile deneyi tekrar başlat; loss eğrilerini karşılaştır.`}</p></article>
          <article><span>04 / GENELLEME</span><h3>Aşırı öğrenme</h3><p>{gap !== null ? `Eğitim−test farkı ${(gap * 100).toFixed(1)} yüzde puan. ${gap > 0.1 ? 'Bu fark aşırı öğrenme işareti olabilir; gürültüyü, L2’yi ve kapasiteyi karşılaştır.' : 'Tek bir fark aşırı öğrenmeyi kanıtlamaz; test kaybının zaman içindeki eğilimini izle.'}` : 'Genellemeyi incelemek için test bölmesini aç. Test noktaları hiçbir zaman ağırlık güncellemesine katılmaz.'}</p></article>
        </div>{tooSmall && <p className="capacity-note">Bu veri kümesi için kapasite yetersiz olabilir. Nöron sayısını artırarak aynı seed ile karşılaştır.</p>}</section>
      </div>{view && trace ? <Inspector snapshot={view} trace={trace} selection={safeSelection} probe={activeProbe} onProbe={setProbe} onSelect={setSelection} config={config} mapImage={images[safeSelection.layer + 1]?.[safeSelection.neuron]} replayEpoch={shownEpoch}/> : <aside className="panel loading-panel">{problem ? 'İnceleyici, eğitilebilir veri olunca açılır.' : 'İnceleyici hazırlanıyor…'}</aside>}</div>
      <div className="bottom-tools"><div><span className="eyebrow">DENEYİ YANINA AL</span><p>Ayarlar tarayıcıda saklanır. Eğitilmiş ağırlıkları saklamak için Model JSON’u indir.<Help label="Dosyalar" text={HELP.files}/></p></div><div className="file-actions"><button onClick={() => exportConfig(config)}>↓ Ayarlar JSON</button><button disabled={!snapshot || pending} onClick={() => snapshot && exportModel(config, snapshot.model, snapshot.epoch)}>↓ Model JSON (ağırlıklar)</button><button onClick={() => fileInput.current?.click()}>↑ JSON içe aktar</button><button disabled={!snapshot} onClick={() => snapshot && download('neuralscope-epoch-gecmisi.csv', historyCSV(snapshot.history), 'text/csv;charset=utf-8')}>↓ Geçmiş CSV</button><input ref={fileInput} aria-label="Deney JSON dosyası" type="file" accept=".json,application/json" hidden onChange={async e => { const f = e.target.files?.[0]; e.target.value = ''; if (!f) return; try {
              if (f.size > 1_000_000) throw Error('Dosya 1 MB’tan küçük olmalı.');
              const file = importExperiment(await f.text());
              const loaded = file.weights ? `Model yüklendi${file.trainedEpochs ? ` (${file.trainedEpochs} epoch eğitilmişti)` : ''}. Ağ kaydedilen ağırlıklardan başlıyor; optimizer belleği ve geçmiş grafiği yeniden başladı.` : 'JSON ayarları yüklendi. Yeni model hazır; önceki ağırlıklar sürdürülmedi.';
              // change() drops loaded weights, so the new ones are set after it. The notice is set again in case the settings were identical.
              change(file.config, loaded); setWeights(file.weights); setReset(n => n + 1); setFileError(''); setNotice(loaded);
            } catch (e) { setFileError(e instanceof Error ? e.message : 'JSON okunamadı.'); } }}/></div></div>
      <footer><span>NeuralScope <span className="footer-dot">·</span> Gerçek hesaplar, görünür öğrenme.</span><span>1. Veri seç <i>→</i> 2. Başlat <i>→</i> 3. Değişimi izle</span></footer>
    </main>
  </div>;
}
