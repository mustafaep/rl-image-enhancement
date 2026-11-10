# Bozuk Görüntüyü Adım Adım İyileştirme: Tablo Tabanlı Pekiştirmeli Öğrenme

*Derin Pekiştirmeli Öğrenme dersi dönem projesi*

> **Durum:** Taslak. Sonuçlar ve tartışma bölümleri deneyler bittikçe doldurulacak.

## Özet
Karanlık ve gürültülü bir fotoğrafı düzeltmek için genelde sabit bir filtre zinciri kullanılır (ör. önce gamma, sonra CLAHE). Ama her fotoğraf aynı zinciri istemez: Biri sadece aydınlatma, diğeri önce gürültü azaltma ister. Bu projede **hangi filtrenin hangi sırayla uygulanacağını** bir pekiştirmeli öğrenme ajanına öğretiyoruz. Ajan görüntünün parlaklık, kontrast ve gürültü düzeyine bakar, 6 klasik işlemden birini seçer ve görüntü aydınlık referansa yaklaştıkça (SSIM artışı) ödül alır. Derin ağ kullanılmaz; ajan **Q-learning, SARSA ve Monte Carlo** ile küçük bir Q-tablosu öğrenir. LOL veri setinde ajanı sabit filtre zincirleri, histogram eşitleme ve rastgele ajanla karşılaştırıyoruz.

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
*(doldurulacak: `results/sonuclar.md` tablosu, öğrenme eğrisi, örnek görüntüler, Q-tablosu ısı haritası)*

## 5. Tartışma
*(doldurulacak: ajan hangi durumda hangi eylemi seçmeyi öğrendi? Q-learning, SARSA, Monte Carlo farkı? Sınırlılıklar: kaba durum tanımı, ödül referans gerektiriyor…)*

## 6. Sonuç

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
