import { useId } from 'react';
/** Plain-language explanations shown behind each "?" button. */
export const HELP = {
  dataset: 'Ağın çözmeye çalışacağı veri şekli. Her nokta iki sayıdan (x₁, x₂) ve bir sınıftan (0 ya da 1) oluşur. XOR, halkalar, aylar ve spiral düz bir çizgiyle ayrılamaz; ağın eğri bir sınır öğrenmesi gerekir. Doğrusal problem tek bir düz çizgiyle ayrılabilir. "Kendi çizimin" ile noktaları sen yerleştirirsin.',
  samples: 'Üretilecek toplam nokta sayısı (40–2000). Çok nokta, şekli daha net gösterir ama her epoch daha uzun sürer.',
  seed: 'Rastgeleliğin başlangıç numarası. Aynı seed her seferinde aynı veriyi ve aynı başlangıç ağırlıklarını üretir; böylece yalnızca bir ayarı değiştirip deneyleri adil karşılaştırabilirsin.',
  noise: 'Noktaların ideal yerlerinden ne kadar saptırılacağı. Gürültü arttıkça sınıflar birbirine karışır ve problem zorlaşır. Noktaların sınıfı değişmez, yalnızca konumu oynar.',
  split: 'Noktaların %20’si kenara ayrılır ve eğitimde hiç kullanılmaz. Ağın daha önce görmediği bu noktalardaki başarısı (test doğruluğu), ezberleyip ezberlemediğini, yani gerçekten öğrenip öğrenmediğini gösterir.',
  features: 'Ağın gördüğü girdiler. Normalde yalnızca x₁ ve x₂ verilir; eğri sınırları ağın kendisi öğrenmek zorundadır. x₁², x₂², x₁·x₂ ya da sin gibi hazır dönüşümler eklersen bazı problemler çok kolaylaşır: örneğin halkalar x₁² ve x₂² ile gizli katman olmadan bile çözülür.',
  layers: 'Giriş ile çıkış arasındaki ara (gizli) katman sayısı. Daha çok katman daha karmaşık şekiller öğrenebilir ama eğitim yavaşlayabilir. 0 seçersen girdiler doğrudan çıkışa bağlanır (lojistik regresyon): bu ağ yalnızca düz bir sınır çizebilir, ek girdi özellikleriyle ise eğri sınırlar da çizebilir.',
  neurons: 'Bu katmandaki nöron (hesap birimi) sayısı. Her nöron girdilerinin ağırlıklı toplamını alıp aktivasyondan geçirir. Nöron azsa ağ karmaşık sınırları çizemeyebilir.',
  activation: 'Her nöronun ağırlıklı toplamı nasıl bükeceği. Bu büküm olmadan ağ yalnızca düz çizgi çizebilirdi. tanh: −1 ile 1 arası yumuşak bir eğri. ReLU: negatif değerleri sıfırlar, pozitifleri olduğu gibi geçirir. Sigmoid: 0 ile 1 arası yumuşak bir eğri; gizli katmanlarda genelde daha yavaş öğrenir.',
  optimizer: 'Hatayı azaltmak için ağırlıkların nasıl güncelleneceği. SGD: her ağırlığı hatayı azaltan yöne, öğrenme hızı kadar iter. Adam: geçmiş adımları hatırlayıp her ağırlığın adım boyunu kendisi ayarlar; genelde daha hızlı ve dengeli öğrenir.',
  learningRate: 'Her güncellemede atılan adımın büyüklüğü (η). Çok küçükse öğrenme çok yavaşlar; çok büyükse kayıp zıplar, hatta ağ bozulur. Adam için 0.001–0.05, SGD için 0.05–1 arası genellikle iyi bir başlangıçtır.',
  batchSize: 'Ağırlıklar her güncellenmeden önce kaç noktanın hatasının ortalamasının alınacağı. Küçük batch: sık ama dalgalı adımlar. Büyük batch: daha seyrek ama daha kararlı adımlar.',
  epochs: 'Bir epoch, eğitim noktalarının hepsinin bir kez kullanılmasıdır. Eğitim bu sayıya ulaşınca kendiliğinden durur.',
  l2: 'Büyük ağırlıklara ceza ekler (λ). Ağı daha sade ve yumuşak sınırlara iter, ezberlemeyi azaltabilir. 0 kapalı demektir; çok büyük değer ağın öğrenmesini engelleyebilir.',
  speed: 'Yalnızca eğitimi ekranda izleme hızını değiştirir. Öğrenmenin sonucunu değiştirmez ve modeli sıfırlamaz.',
  metricEpoch: 'Tüm eğitim verisinin kaç kez baştan sona kullanıldığı. Alttaki ince çubuk, şu anki epoch’un ne kadarının tamamlandığını gösterir.',
  metricLoss: 'Eğitim noktalarındaki ortalama hata (BCE, ikili çapraz entropi). Ağ yanlış ve kendinden emin tahmin yaptıkça büyür; 0’a yaklaşması iyi öğrenme demektir.',
  metricAccuracy: 'Eğitim noktalarının yüzde kaçının doğru sınıfa atandığı. Olasılık 0.5 ve üstüyse tahmin sınıf 1 sayılır.',
  metricTest: 'Eğitimde hiç kullanılmayan test noktalarındaki doğruluk. Ağın daha önce görmediği veride ne kadar başarılı olduğunu, yani genelleme yeteneğini gösterir.',
  metricTime: 'Eğitimin çalıştığı toplam süre. Duraklatılan süre sayılmaz.',
  decision: 'Arka plan, ağın her koordinat için verdiği “sınıf 1” olasılığıdır: mavi tonlar sınıf 0’a, turuncu tonlar sınıf 1’e yakındır. Kesikli çizgi, olasılığın tam 0.5 olduğu karar sınırıdır. Dolu şekiller eğitim, içi boş şekiller test noktalarıdır. Bir yere tıklarsan ağın o noktayı nasıl hesapladığını inceleyicide görürsün.',
  custom: 'Karar sınırına tıklayarak kendi noktalarını yerleştir. Önce hangi sınıfı ekleyeceğini seç; “Sil” ile tıkladığın yere en yakın noktayı kaldırırsın. Her değişiklikte model baştan kurulur. “İncele” moduna geçince tıklamak yine inceleme noktasını seçer.',
  network: 'Sütunlar katmanlardır: solda girdiler, ortada gizli katmanlar, sağda çıkış. Çizgiler ağırlıklardır: mavi pozitif, turuncu negatif; kalınlık büyüklüğü gösterir. Bir nörona ya da çizgiye tıklarsan ayrıntısı inceleyicide açılır.',
  maps: 'Her kare, o nöronun düzlemin her noktasında ürettiği değeri gösterir: mavi pozitif, turuncu negatif, koyu sıfıra yakın. İlk katmandaki nöronlar genelde basit, düz geçişler öğrenir; sonraki katmanlar bunları birleştirip daha karmaşık şekiller oluşturur. Çıkış karesi, büyük karar sınırının aynısıdır.',
  timeline: 'Eğitim sırasında ağırlıkların kopyaları saklanır. Kaydırıcıyla geçmiş bir epoch’a gidip karar sınırının, ağın ve nöronların o andaki halini görebilir, ▶ ile öğrenmeyi baştan oynatabilirsin. Eğitime devam edince canlı görünüme dönülür.',
  charts: 'Kayıp (BCE), tahminlerin ne kadar yanlış olduğunu ölçer; düşmesi öğrenme demektir. Doğruluk, doğru sınıflandırılan noktaların oranıdır. Düz çizgi eğitim, kesikli çizgi test verisidir. Eğitim kaybı düşerken test kaybı yükseliyorsa ağ ezberliyor olabilir.',
  compare: 'Önceki deneyin eğitim eğrileri gri çizgi olarak kalır; böylece değiştirdiğin ayarın etkisini aynı grafikte görürsün. Üstteki etikette neyin değiştiği yazar.',
  inspectorForward: 'Seçili nöron, gelen her değeri (aᵢ) kendi ağırlığıyla (wᵢ) çarpar, bunları toplar ve bias’ı (b) ekler: bu z’dir. Sonra z aktivasyon fonksiyonundan geçer ve nöronun çıktısı a olur.',
  inspectorBackward: 'Geri yayılım, hatanın bu nörona ve ağırlığa ne kadar bağlı olduğunu hesaplar (türev). Türev pozitifse ağırlığı azaltmak hatayı düşürür, negatifse artırmak düşürür.',
  inspectorUpdate: 'Son eğitim adımında bu ağırlığa gerçekten uygulanan değişiklik. Batch’teki tüm noktaların ortalama türevine göre optimizer’ın attığı adımdır.',
  inspectorMap: 'Seçili nöronun düzlemin her noktasındaki çıktısı. Mavi pozitif, turuncu negatif, koyu sıfıra yakın değerdir. Bu nöronun “neye tepki verdiğini” gösterir.',
  files: 'Ayarlar JSON’u yalnızca ayarları kaydeder. Model JSON’u bunlara ek olarak o anki ağırlıkları da kaydeder; içe aktarınca ağ kaldığı ağırlıklardan başlar. Optimizer belleği ve geçmiş grafiği yeniden başlar. Geçmiş CSV tüm epoch ölçümlerini tablo olarak indirir.',
};
/** "?" button whose explanation opens on hover, tap or keyboard focus. */
export function Help({ label, text }: { label: string; text: string }) {
  const id = useId();
  return <span className="help"><button type="button" className="help-button" aria-label={`${label} nedir?`} aria-describedby={id}
    onClick={e => { e.preventDefault(); e.currentTarget.focus(); }} onKeyDown={e => { if (e.key === 'Escape') e.currentTarget.blur(); }}>?</button>
    <span role="tooltip" id={id} className="help-tip">{text}</span></span>;
}
