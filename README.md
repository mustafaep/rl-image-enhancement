# Bozuk Görüntüyü Adım Adım İyileştirme: Tablo Tabanlı Pekiştirmeli Öğrenme

*Derin Pekiştirmeli Öğrenme dersi dönem projesi*

## Özet
Karanlık ve gürültülü görüntülerin iyileştirilmesinde genellikle gamma düzeltme ve CLAHE gibi işlemler belirli bir sırayla uygulanır. Ancak bu işlemlerin her görüntüde aynı sonucu vermesi beklenemez. Bu projede, görüntünün özelliklerine göre uygun iyileştirme işlemini ve uygulama sırasını seçebilen bir pekiştirmeli öğrenme ajanı geliştiriyoruz.

Ajan, görüntünün parlaklık, kontrast ve gürültü seviyelerini değerlendirerek altı eylemden birini seçiyor: beş klasik görüntü işleme yöntemi ve "dur". Elde edilen sonuç, aydınlık referans görüntüyle SSIM üzerinden karşılaştırılıyor ve iyileşme miktarına göre ajan ödüllendiriliyor. Projede derin öğrenme ağı yerine Q-learning, SARSA ve Monte Carlo algoritmalarıyla bir Q-tablosu oluşturuluyor. Geliştirilen ajanların performansı LOL veri seti üzerinde sabit bir filtre zinciri (Gamma + CLAHE), histogram eşitleme ve rastgele seçim yapan bir ajanla karşılaştırılıyor. Q-learning ve SARSA ajanları, SSIM'de en iyi sabit yöntemi yaklaşık 0,10 farkla geçiyor (0,71'e karşı 0,61).

## 1. Giriş
- Problem: düşük ışıkta çekilmiş görüntüleri iyileştirme
- Neden sıralı karar problemi? Filtrelerin sırası sonucu değiştirir (ör. önce aydınlatıp sonra gürültü azaltmak ≠ tersi)
- Neden tablo tabanlı RL? Durum uzayı küçük tutulabilir, öğrenilen politika tablodan okunup yorumlanabilir

## 2. Yöntem

### 2.1 Markov karar süreci (MDP)

| Bileşen | Tanım |
|---|---|
| Durum *s* | (parlaklık, kontrast, gürültü) özellikleri aralıklara bölünür: 5 × 4 × 3 = **60 durum** |
| Eylem *a* | `aydinlat` (γ=0,6), `karart` (γ=1,25), `clahe`, `gurultu_azalt` (bilateral), `keskinlestir` (unsharp mask), `dur` |
| Ödül *r* | 10 · (SSIM<sub>sonra</sub> − SSIM<sub>önce</sub>) − 0,01 (adım cezası) |
| Bölüm sonu | `dur` eylemi ya da 5 adım |

### 2.2 Algoritmalar
- **Q-learning:** Q(s,a) ← Q(s,a) + α [r + γ max<sub>a'</sub> Q(s',a') − Q(s,a)]
- **SARSA:** Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') − Q(s,a)], burada a' gerçekten seçilen eylem
- **Monte Carlo:** Bölüm bitince her adım için gerçek getiri G hesaplanır, Q(s,a) ← Q(s,a) + α [G − Q(s,a)]
- Hepsinde ε-açgözlü keşif (ε: 1,0 → 0,05), α = 0,1, γ = 0,95

### 2.3 Karşılaştırma yöntemleri
Girdi (hiçbir şey yapmamak) · Rastgele ajan · Histogram eşitleme · Sabit Gamma + CLAHE · Aç gözlü kâhin (her adımda referansa bakarak en iyi eylemi seçer; gerçekte yapılamaz, **üst sınır** olarak verilir)

## 3. Deney düzeni
- Veri: LOL (Wei ve ark., 2018): 485 eğitim, 15 test çifti (600×400). Eğitim hız için 128×128'de yapıldı. Karanlık görüntülerin yarısına yapay Gauss gürültüsü eklendi.
- Tam çözünürlük: Durum her zaman 128×128'lik bir kopyadan hesaplandığı ve filtre parametreleri görüntü boyutuyla orantılı ölçeklendiği için, 128×128'de eğitilen Q-tablosu orijinal 600×400 görüntülere doğrudan uygulanabilir (`evaluate.py --tam`).
- Ölçütler: PSNR, SSIM, ortalama adım sayısı
- Eğitim: 3000 bölüm

## 4. Sonuçlar

### 4.1 Test seti (128×128)
LOL eval15, 15 görüntü. Ajanlar 3000 bölüm eğitildi; testte keşif kapalı (ε = 0).

| Yöntem | PSNR ↑ | SSIM ↑ | Ort. adım |
|---|---|---|---|
| Girdi (işlem yok) | 7,83 | 0,161 | 0,0 |
| Rastgele ajan | 9,84 | 0,335 | 3,1 |
| Histogram eşitleme | 13,91 | 0,506 | 1,0 |
| Sabit Gamma + CLAHE | 14,70 | 0,610 | 2,0 |
| **RL: Q-learning** | 15,91 | 0,713 | 3,8 |
| **RL: SARSA** | **16,33** | **0,714** | 3,6 |
| **RL: Monte Carlo** | 14,96 | 0,682 | 2,9 |
| Açgözlü kâhin (üst sınır) | 18,83 | 0,754 | 3,9 |

Üç RL ajanı da en iyi sabit yöntemi (Gamma + CLAHE) geçiyor: SSIM'de Q-learning ve SARSA +0,10, Monte Carlo +0,07. Açgözlü kâhin her adımda referans görüntüye bakarak en iyi eylemi seçtiği için gerçekte uygulanamaz; ulaşılabilecek üst sınırı göstermek için verildi.

### 4.2 Orijinal çözünürlük (600×400)
Aynı Q-tabloları, yeniden eğitilmeden orijinal boyuttaki görüntülere uygulandı (`evaluate.py --tam`).

| Yöntem | PSNR ↑ | SSIM ↑ | Ort. adım |
|---|---|---|---|
| Girdi (işlem yok) | 7,77 | 0,173 | 0,0 |
| Rastgele ajan | 9,47 | 0,276 | 3,1 |
| Histogram eşitleme | 13,98 | 0,334 | 1,0 |
| Sabit Gamma + CLAHE | 13,69 | 0,440 | 2,0 |
| **RL: Q-learning** | 15,55 | **0,575** | 3,7 |
| **RL: SARSA** | **15,64** | 0,558 | 3,5 |
| **RL: Monte Carlo** | 14,30 | 0,526 | 2,8 |
| Açgözlü kâhin (üst sınır) | 17,46 | 0,692 | 4,5 |

Bütün yöntemlerin SSIM'i orijinal boyutta düşüyor, ama sıralama korunuyor ve RL ile sabit yöntem arasındaki fark büyüyor (Q-learning +0,135).

### 4.3 Öğrenme eğrisi

![Öğrenme eğrisi](results/ogrenme_egrisi.png)

500 bölümlük dilimlerde ortalama toplam ödül:

| Bölüm | 1–500 | 501–1000 | 1001–1500 | 1501–2000 | 2001–2500 | 2501–3000 |
|---|---|---|---|---|---|---|
| Q-learning | 2,51 | 4,23 | 4,33 | 4,79 | 4,66 | 4,93 |
| SARSA | 2,43 | 4,29 | 4,62 | 4,70 | 4,91 | 4,95 |
| Monte Carlo | 2,33 | 4,14 | 4,04 | 4,17 | 4,48 | 4,80 |

Ödül ilk 1000 bölümde hızla artıyor, sonra yavaşça düzleşiyor. ε yaklaşık 1500. bölümde alt sınırı olan 0,05'e iniyor. Monte Carlo en yavaş yükselen yöntem.

### 4.4 Öğrenilen politika (Q-tablosu)

![Q-tablosu: Q-learning](results/q_tablosu_qlearning.png)

60 durumdan 48'i eğitim sırasında ziyaret edildi. Karanlık durumlarda (parlaklık aralığı 0–1, durum 0–23) en yüksek değer çoğunlukla **aydınlat** eyleminde. Aydınlık durumlarda (durum 36 ve sonrası) filtrelerin değeri çoğunlukla 0'a yakın ya da negatif, bu yüzden ajan çoğunlukla **dur** eylemini seçiyor; birkaç durumda (ör. 44) küçük pozitif değerle gürültü azaltmayı deniyor. Test setinde en sık eylem dizileri:

| Ajan | 128×128 | 600×400 |
|---|---|---|
| Q-learning | gürültü azalt → aydınlat → aydınlat → dur | aydınlat → aydınlat → dur |
| SARSA | aydınlat → aydınlat → dur | aydınlat → aydınlat → dur |
| Monte Carlo | gürültü azalt → aydınlat → aydınlat → dur | aydınlat → aydınlat → dur |
| Sabit yöntem | aydınlat → CLAHE (her görüntüde aynı) | aydınlat → CLAHE |

### 4.5 Örnek görüntüler

![Örnekler (128×128)](results/ornekler.png)

Orijinal çözünürlükte, ölçeklenmeden (girdi · Gamma + CLAHE · Q-learning · SARSA · referans):

![eval15/1.png](results/tam/1.png)

Bütün test görüntüleri için: [`results/tam/`](results/tam/)

## 5. Tartışma

**Ajan ne öğrendi?** Q-tablosuna bakınca ajanın öğrendiği şey aslında oldukça sezgisel. Görüntü karanlıkken neredeyse her durumda en yüksek değer "aydınlat" eyleminde. Karanlık ve gürültülü durumlarda (ör. durum 2) önce gürültü azaltmayı tercih ediyor; bu da mantıklı, çünkü önce aydınlatmak gürültüyü de büyütüyor. Orta parlaklıktaki bazı durumlarda (12, 13) CLAHE öne çıkıyor. Görüntü yeterince aydınlandıktan sonra ise filtrelerin değeri sıfıra yakın ya da negatif olduğu için ajan duruyor. Yani ajan sadece "ne yapacağını" değil, "ne zaman bırakacağını" da öğrenmiş. Sabit Gamma + CLAHE zinciri bunu yapamıyor: her görüntüye aynı iki adımı uyguluyor.

**Üç yöntemin karşılaştırması.** Q-learning ve SARSA neredeyse aynı sonucu verdi (SSIM 0,713 ve 0,714). Bunu bekliyorduk: bölümler çok kısa (en fazla 5 adım) ve ε eğitimin ikinci yarısında 0,05'e indiği için iki yöntemin hedefleri arasındaki fark küçülüyor. Monte Carlo biraz geride kaldı (0,682) ve öğrenme eğrisi de daha yavaş yükseldi. Monte Carlo güncelleme için bölümün bitmesini bekliyor ve hedef olarak o bölümde gerçekten toplanan getiriyi kullanıyor. Bu getiri seçilen görüntüye ve keşif sırasında yapılan rastgele hamlelere çok bağlı, yani gürültülü. Aynı bölüm sayısında tablosu daha geç oturuyor.

**Ajan ile kâhin arasındaki fark.** Kâhin her adımda referansa bakarak seçim yaptığı için 0,754'e ulaşıyor; ajan 0,713'te kalıyor. Farkın ana nedeni durum tanımının kaba olması. 60 durum, birbirinden oldukça farklı görüntüleri aynı numaraya koyuyor. Örneğin eval15/1.png'de ajan 44 numaralı durumda gürültü azaltmayı seçti, çünkü o duruma düşen diğer görüntülerde bu işe yaramıştı; bu görüntüde ise SSIM hafifçe düştü. Eğitimi uzatmak bu farkı kapatmıyor: ödül 2000. bölümden sonra zaten düzleşiyor. Farkı kapatmanın yolu durumu zenginleştirmek (daha fazla aralık ya da yeni özellikler), ama bu tabloyu büyütüyor ve her hücrenin öğrenilmesi için daha çok bölüm gerekiyor.

**Çözünürlük.** Ajan 128×128'de eğitildi ama orijinal 600×400 görüntülerde de sabit yöntemi geçti (0,575'e karşı 0,440). Orijinal boyutta bütün yöntemlerin SSIM'i düşüyor, çünkü küçültme gürültüyü yumuşatıyor; büyük görüntüde gürültü olduğu gibi duruyor. Kâhinle fark da orijinal boyutta büyüyor (0,575'e karşı 0,692). Bunun bir nedeni, ajanın durumu küçültülmüş kopyadan okuması ve büyük görüntüdeki gürültüyü olduğundan az görmesi olabilir.

**Sınırlılıklar.**
- Ödül için aydınlık referans görüntü gerekiyor. Eğitilmiş ajan referansa ihtiyaç duymuyor (sadece durumu kullanıyor), ama eğitim için aynı sahnenin karanlık ve aydınlık çiftleri şart.
- SSIM insan gözüyle birebir örtüşmüyor. SSIM'i yüksek bir sonuç göze her zaman en iyi görünen sonuç olmayabilir.
- Eylemler sabit ayarlı (ör. gamma hep 0,6). Ajan filtrenin gücünü ayarlayamıyor, sadece aynı filtreyi tekrar uygulayabiliyor.
- Durum tüm görüntünün ortalamasına bakıyor. Bir köşesi aydınlık, diğer köşesi karanlık bir görüntüyü ayırt edemiyor.
- Sadece LOL veri setinde denendi (iç mekân sahneleri); test seti 15 görüntü ve tek bir rastgele tohumla çalıştırıldı.

## 6. Sonuç
Bu projede karanlık görüntülere hangi filtrenin hangi sırayla uygulanacağını, 60 durumlu ve 6 eylemli küçük bir Q-tablosuyla öğrenen bir ajan geliştirdik. Derin ağ kullanmadan, yaklaşık 30 saniyelik bir eğitimle ajan, her görüntüye aynı işlemi uygulayan Gamma + CLAHE zincirini hem 128×128'de (SSIM 0,71'e karşı 0,61) hem de orijinal çözünürlükte (0,58'e karşı 0,44) geçti. Öğrenilen politika tablodan doğrudan okunabiliyor: karanlıkta aydınlat, gürültü varsa önce onu azalt, yeterince iyiyse dur. Yöntemin sınırı durum tanımının kabalığı; ileride durumun zenginleştirilmesi, ayarlanabilir eylemler ve referans gerektirmeyen bir ödül denenebilir.

## Çalıştırma

```bash
pip install -r requirements.txt
# LOL veri setini data/LOL/ altına koy (bkz. data/README.md)
python train.py --ajan hepsi --bolum 3000
python evaluate.py          # 128×128
python evaluate.py --tam    # orijinal 600×400; results/tam/ altında piksel piksel karşılaştırmalar
```

Çıktılar `results/` klasörüne yazılır.

## Klasör yapısı

```
src/env.py        ortam: durum, eylemler, ödül
src/agents.py     Q-learning, SARSA, Monte Carlo
src/baselines.py  karşılaştırma yöntemleri
src/data.py       LOL yükleme
train.py          eğitim
evaluate.py       test + tablolar + görseller
notebooks/        adım adım anlatım
```

## Kaynaklar
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2. baskı). MIT Press.
- Wei, C., Wang, W., Yang, W., & Liu, J. (2018). Deep Retinex Decomposition for Low-Light Enhancement. *BMVC*.
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279–292.
