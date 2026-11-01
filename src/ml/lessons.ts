import type { Config, EpochMetric } from '../types';
import { DEFAULT, validateConfig } from './config';

/** Lesson texts are built only from the measured values of a finished run. */
export interface StepOutcome {
  expected: boolean;
  text: string;
}
export interface LessonStep {
  title: string;
  task: string;
  look: string;
  setup: Partial<Config>;
  explain: (history: EpochMetric[]) => StepOutcome;
}
export interface Lesson {
  id: string;
  title: string;
  summary: string;
  base: Partial<Config>;
  conclusion: string;
  steps: LessonStep[];
}

const pct = (n: number | null) => (n === null ? '—' : `%${(n * 100).toFixed(1)}`);
const num = (n: number | null) => (n === null ? '—' : n.toFixed(4));
const summary = (h: EpochMetric[]) =>
  `Eğitim doğruluğu ${pct(h.at(-1)!.trainAccuracy)}, test doğruluğu ${pct(h.at(-1)!.testAccuracy)}.`;
const outcome = (expected: boolean, text: string, h: EpochMetric[]): StepOutcome => ({
  expected,
  text: expected
    ? text
    : `Bu sefer beklenen sonuç çıkmadı: ${summary(h)} Eğitim böyle de bitebilir; aynı seed ile yeniden kurup deneyebilirsin.`,
});

export const LESSONS: Lesson[] = [
  {
    id: 'learning-rate',
    title: 'Öğrenme hızı',
    summary: 'Aynı ağ, üç adım boyu: biri öğrenir, biri savrulur, biri ağır ağır ilerler.',
    base: { dataset: 'xor', hidden: [8, 8], optimizer: 'adam', epochs: 200 },
    conclusion:
      'Öğrenme hızı, ağın her güncellemede ne kadar ilerleyeceğini belirler. Çok büyükse iyi noktanın üstünden atlar, çok küçükse oraya varması uzun sürer. Doğru değer probleme ve optimizer’a göre değişir; bu yüzden denenerek bulunur.',
    steps: [
      {
        title: 'Uygun bir adım boyu',
        task: 'Öğrenme hızı η = 0.02. Ağı eğit; bu deneme karşılaştırma noktamız olacak.',
        look: 'Kayıp grafiğinin ne kadar hızlı düştüğüne bak.',
        setup: { learningRate: 0.02 },
        explain: h => {
          const first = h[0],
            last = h.at(-1)!;
          return outcome(
            last.trainLoss < 0.05 && last.trainAccuracy >= 0.95,
            `Kayıp ${num(first.trainLoss)} değerinden ${num(last.trainLoss)} değerine indi. ${summary(h)} Ağ bu adım boyuyla rahatça öğrendi.`,
            h,
          );
        },
      },
      {
        title: 'Çok büyük adımlar',
        task: 'Öğrenme hızı 50 katına çıktı: η = 1. Aynı ağı aynı başlangıçtan yeniden eğit.',
        look: 'Yeşil çizgiyi önceki denemenin gri çizgisiyle karşılaştır.',
        setup: { learningRate: 1 },
        explain: h => {
          const last = h.at(-1)!,
            tail = h.slice(-20).map(m => m.trainLoss);
          return outcome(
            last.trainAccuracy < 0.8,
            `Eğitim doğruluğu ${pct(last.trainAccuracy)} seviyesinde kaldı. Son 20 epoch’ta kayıp ${num(Math.min(...tail))} ile ${num(Math.max(...tail))} arasında gidip geldi: adımlar o kadar büyük ki ağ iyi bir noktaya oturamıyor.`,
            h,
          );
        },
      },
      {
        title: 'Çok küçük adımlar',
        task: 'Şimdi tersi: η = 0.0005, ilk denemenin 40’ta biri. Yine aynı ağı eğit.',
        look: 'Kayıp eğrisinin eğimine ve son değerine bak.',
        setup: { learningRate: 0.0005 },
        explain: h => {
          const last = h.at(-1)!;
          return outcome(
            last.trainLoss > 0.1,
            `200 epoch sonunda kayıp hâlâ ${num(last.trainLoss)}; ilk denemedeki η = 0.02 ile çok daha aşağı inmişti. ${summary(h)} Yön doğru ama adımlar küçük olduğu için ağ aynı yolu çok daha yavaş alıyor.`,
            h,
          );
        },
      },
    ],
  },
  {
    id: 'capacity',
    title: 'Ağın kapasitesi',
    summary: 'Halkaları ayırmak için kaç nöron gerekir? Karar sınırı cevabı gösteriyor.',
    base: { dataset: 'circles', epochs: 300 },
    conclusion:
      'Her nöron kabaca düz bir kenar öğrenir, sonraki katman bu kenarları birleştirir. Kenar sayısı şekli çevrelemeye yetmezse eğitim ne kadar uzarsa uzasın sonuç değişmez. Üç nöron bu problem için sınırdadır: seed 42’de takılır, 1, 2, 3 ve 7 seed’lerinde çözer. Seed alanından kendin de deneyebilirsin.',
    steps: [
      {
        title: 'Gizli katman yok',
        task: 'Girdiler doğrudan çıkışa bağlı. Bu ağ yalnızca tek bir düz sınır çizebilir. Eğit ve sonucu gör.',
        look: 'Karar sınırı panelindeki kesikli çizgiye bak.',
        setup: { hidden: [] },
        explain: h =>
          outcome(
            h.at(-1)!.trainAccuracy < 0.7,
            `Eğitim doğruluğu yalnızca ${pct(h.at(-1)!.trainAccuracy)}. Tek düz çizgi bir halkayı içeriden ayıramaz; yazı turadan pek farkı yok.`,
            h,
          ),
      },
      {
        title: 'İki nöron',
        task: 'Araya iki nöronlu bir katman eklendi. Ağ artık iki kenar kullanabilir.',
        look: 'Karar sınırının şekline ve ağ panelindeki nöron haritalarına bak.',
        setup: { hidden: [2] },
        explain: h =>
          outcome(
            h.at(-1)!.trainAccuracy >= 0.7 && h.at(-1)!.trainAccuracy < 0.98,
            `${summary(h)} Tek çizgiden çok daha iyi ama yetmiyor: iki kenarla iç halkanın etrafı kapatılamıyor.`,
            h,
          ),
      },
      {
        title: 'Dört nöron',
        task: 'Nöron sayısı dörde çıktı. Aynı veriyi yeniden eğit.',
        look: 'Karar sınırının iç halkanın çevresini kapatıp kapatmadığına bak.',
        setup: { hidden: [4] },
        explain: h =>
          outcome(
            h.at(-1)!.trainAccuracy >= 0.98,
            `${summary(h)} Dört kenar iç halkayı çevrelemeye yetti. Nöron haritalarında her nöronun farklı bir yönde kenar öğrendiğini görebilirsin.`,
            h,
          ),
      },
    ],
  },
  {
    id: 'features',
    title: 'Girdi özellikleri',
    summary: 'Daha büyük ağ yerine veriyi doğru göstermek de problemi çözebilir.',
    base: { dataset: 'circles', hidden: [], epochs: 200 },
    conclusion:
      'Halkanın denklemi x₁² + x₂² = r² olduğu için problem, yeni girdilerde düz bir çizgiyle ayrılır hale geldi. Sinir ağlarının gizli katmanları da benzer bir iş yapar: veriyi kolay ayrılacak bir biçime dönüştürmeyi öğrenir. Burada dönüşümü biz hazır verdik.',
    steps: [
      {
        title: 'Yalnızca x₁ ve x₂',
        task: 'Gizli katman yok, girdiler ham koordinatlar. Önceki dersten bildiğimiz gibi bu ağ halkaları ayıramamalı.',
        look: 'Doğruluk metriğine bak.',
        setup: { features: ['x1', 'x2'] },
        explain: h =>
          outcome(
            h.at(-1)!.trainAccuracy < 0.7,
            `Eğitim doğruluğu ${pct(h.at(-1)!.trainAccuracy)}. Ham koordinatlarla tek düz sınır işe yaramadı.`,
            h,
          ),
      },
      {
        title: 'x₁² ve x₂²',
        task: 'Ağ aynı derecede basit: gizli katman yok. Yalnızca girdiler değişti, artık koordinatların kareleri veriliyor.',
        look: 'Karar sınırına bak: düz çizgi çizen ağ şimdi ne çiziyor?',
        setup: { features: ['x1sq', 'x2sq'] },
        explain: h =>
          outcome(
            h.at(-1)!.trainAccuracy >= 0.95,
            `${summary(h)} Tek bir nöron, doğru girdilerle problemi çözdü. Karar sınırı koordinat düzleminde bir daire gibi görünür, ama ağ açısından hâlâ düz bir sınırdır.`,
            h,
          ),
      },
    ],
  },
  {
    id: 'overfitting',
    title: 'Aşırı öğrenme',
    summary: 'Az ve gürültülü veriyle büyük ağ öğrenmek yerine ezberler. İki çareyi dene.',
    base: {
      dataset: 'circles',
      samples: 100,
      noise: 0.2,
      batchSize: 16,
      epochs: 800,
      speed: 'fast',
    },
    conclusion:
      'Eğitim başarısı tek başına yeterli değildir; asıl ölçü ağın görmediği veridir. L2 cezası ve daha küçük bir ağ ezberlemeyi azaltmanın iki yoludur. Bu derste test kümesi yalnızca 20 nokta olduğu için bir nokta doğruluğu 5 puan oynatır; bu yüzden test kaybı doğruluktan daha güvenilir bir işarettir.',
    steps: [
      {
        title: 'Büyük ağ, az veri',
        task: '100 gürültülü nokta ve üç katmanda 16’şar nöron. Ağın ezberlemek için fazlasıyla yeri var. Eğit.',
        look: 'Kayıp grafiğinde yeşil (eğitim) ve turuncu (test) çizginin ayrılmasına bak.',
        setup: { hidden: [16, 16, 16], l2: 0 },
        explain: h => {
          const last = h.at(-1)!,
            best = h.reduce((b, m) => ((m.testLoss ?? Infinity) < (b.testLoss ?? Infinity) ? m : b));
          return outcome(
            last.trainAccuracy - (last.testAccuracy ?? 0) >= 0.1 &&
              (last.testLoss ?? 0) > 1.5 * (best.testLoss ?? 0),
            `${summary(h)} Test kaybı epoch ${best.epoch}’te ${num(best.testLoss)} ile en düşük noktasını gördü, sonra ${num(last.testLoss)} değerine yükseldi. Eğitim kaybı ${num(last.trainLoss)} iken bu, ağın genel kuralı değil tek tek noktaları öğrendiğini gösterir. Geri sarıp karar sınırının nasıl karmaşıklaştığını izleyebilirsin.`,
            h,
          );
        },
      },
      {
        title: 'Çare 1: L2 cezası',
        task: 'Aynı büyük ağ, ama büyük ağırlıklar artık cezalı: L2 = 0.01.',
        look: 'Turuncu test çizgisinin artık yukarı kaçıp kaçmadığına ve karar sınırının yumuşaklığına bak.',
        setup: { hidden: [16, 16, 16], l2: 0.01 },
        explain: h => {
          const last = h.at(-1)!;
          return outcome(
            (last.testLoss ?? Infinity) < 0.4,
            `${summary(h)} Test kaybı bu sefer ${num(last.testLoss)} oldu. Eğitim kaybı ${num(last.trainLoss)} ile daha yüksek kaldı: ceza, ağın noktaları tek tek ezberlemesini zorlaştırdı.`,
            h,
          );
        },
      },
      {
        title: 'Çare 2: Küçük ağ',
        task: 'Ceza yok, ama ağ küçüldü: tek katmanda 4 nöron. Ezberlemeye yeri kalmadı.',
        look: 'Karar sınırının sadeliğine ve test kaybına bak.',
        setup: { hidden: [4], l2: 0 },
        explain: h => {
          const last = h.at(-1)!;
          return outcome(
            (last.testLoss ?? Infinity) < 0.4 && (last.testAccuracy ?? 0) >= 0.85,
            `${summary(h)} Test kaybı ${num(last.testLoss)}. Parametre sayısı çok azaldı ama görmediği veride büyük ağdan daha iyi sonuç verdi.`,
            h,
          );
        },
      },
    ],
  },
];

/** The exact experiment a step runs. Drawn points are carried along so a lesson never deletes them. */
export function stepConfig(lesson: Lesson, step: LessonStep, current: Pick<Config, 'customPoints'>): Config {
  return validateConfig({
    ...DEFAULT,
    speed: 'normal',
    ...lesson.base,
    ...step.setup,
    customPoints: current.customPoints,
  });
}
