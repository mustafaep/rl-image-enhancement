"""
İkinci deney: ödülü OCR başarısı olan görüntü iyileştirme ortamı.

Fikir: Ajan karanlık, bazen bulanık ve/veya gürültülü bir YAZI görüntüsüyle
başlar. Her adımda bir OpenCV filtresi ve parametresini seçer (ör. "gamma 0,5"
ya da "CLAHE clip 4,0"). Tesseract OCR görüntüdeki yazıyı ne kadar doğru
okursa o kadar ödül alır. OCR motoru ajanın değil ORTAMIN parçasıdır:
ödülü o üretir, ajan yalnız Q-tablosuyla karar verir.

Veri sentetiktir: rastgele bir "kelime + 4 rakam" düz zemine yazılır, sonra
karartılır, bulanıklaştırılır, gürültü eklenir. Dosya saklanmaz; aynı tohumla
her seferinde aynı görüntüler üretilir.

Durum: parlaklık (4) x kontrast (2) x gürültü (3) x keskinlik (3) = 72 durum.
Arayüz src/env.py'deki GoruntuOrtami ile aynı, böylece src/agents.py'deki
ajanlar hiç değişmeden kullanılır.
"""
import os
os.environ.setdefault("OMP_THREAD_LIMIT", "1")   # Tesseract tek çekirdek kullansın (paralel eğitim için)
import re
import numpy as np
import cv2
import pytesseract

# ---------------------------------------------------------------- veri
GEN, YUK = 320, 64                                  # görüntü boyutu (piksel)
HARFLER = "ABCDEFGHJKLMNPRSTUVWXYZ"                 # O, I, Q yok (0, 1, O ile karışmasın)
RAKAMLAR = "23456789"                               # 0 ve 1 yok
YAZI_TIPLERI = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
                cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX]
TUR_ADLARI = ["yalnız karanlık", "karanlık + bulanık", "karanlık + gürültülü", "hepsi"]


def metin_uret(rng):
    kelime = "".join(rng.choice(list(HARFLER), int(rng.integers(5, 8))))
    sayi = "".join(rng.choice(list(RAKAMLAR), 4))
    return f"{kelime} {sayi}"


def temiz_goruntu(metin, rng):
    """Açık düz zemin, koyu yazı. 3 kanallı float32, 0-1."""
    zemin = rng.uniform(0.75, 0.95)
    yazi_rengi = rng.uniform(0.0, 0.25)
    font = YAZI_TIPLERI[int(rng.integers(len(YAZI_TIPLERI)))]
    kalinlik = int(rng.integers(1, 3))
    olcek = rng.uniform(0.75, 1.15)
    while True:                                     # yazı çerçeveye sığana kadar küçült
        (w, h), alt = cv2.getTextSize(metin, font, olcek, kalinlik)
        if w <= GEN - 16 and h + alt <= YUK - 10:
            break
        olcek *= 0.95
    x = int(rng.integers(8, GEN - w - 7))
    y = int(rng.integers(h + 5, YUK - alt - 4))     # y = yazının taban çizgisi
    img = np.full((YUK, GEN), round(zemin * 255), np.uint8)       # putText 8 bit ister
    cv2.putText(img, metin, (x, y), font, olcek, round(yazi_rengi * 255), kalinlik, cv2.LINE_AA)
    return np.dstack([img] * 3).astype(np.float32) / 255


def boz(img, rng):
    """Bulanıklık (optik) -> karartma -> gürültü (sensör) -> 8 bit'e yuvarlama (kamera)."""
    sigma = 0.0 if rng.random() < 0.4 else rng.uniform(0.3, 2.0)
    std = 0.0 if rng.random() < 0.4 else rng.uniform(0.005, 0.05)
    carpan = rng.uniform(0.08, 0.40)
    if sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    img = img * carpan
    if std > 0:
        img = img + rng.normal(0, std, img.shape[:2])[:, :, None]
    img = np.round(np.clip(img, 0, 1) * 255) / 255
    tur = (sigma > 0) + 2 * (std > 0)               # 0 karanlık, 1 +bulanık, 2 +gürültülü, 3 hepsi
    return img.astype(np.float32), int(tur)


def veri_uret(n, tohum):
    """n örnek: [(bozuk, temiz, metin, tür), ...]. Tesseract'ın temiz hâlini doğru
    okuyamadığı örnekler elenir (ödül ölçülemez olurdu). Tohumla belirlenimci."""
    rng = np.random.default_rng(tohum)
    ornekler, elenen = [], 0
    while len(ornekler) < n:
        metin = metin_uret(rng)
        temiz = temiz_goruntu(metin, rng)
        if ocr_skoru(temiz, metin) < 1.0:
            elenen += 1
            continue
        bozuk, tur = boz(temiz, rng)
        ornekler.append((bozuk, temiz, metin, tur))
    return ornekler, elenen


# ---------------------------------------------------------------- OCR (ortamın parçası)
# psm 7 = tek satır yazı. Beyaz listede boşluk da var (yoksa Tesseract kelimeleri birleştiriyor).
OCR_AYARI = '--psm 7 -l eng -c "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "'


def ocr_oku(img):
    gri = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    try:
        return pytesseract.image_to_string(gri, config=OCR_AYARI)
    except pytesseract.TesseractError:
        return ""   # Tesseract bazı çok bozuk görüntülerde çöküyor: "hiçbir şey okunamadı" say


def _duzelt(s):
    return re.sub(r"\s+", " ", s.upper()).strip()


def levenshtein(a, b):
    """a'yı b'ye çevirmek için gereken en az ekleme/silme/değiştirme sayısı (DP)."""
    onceki = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        simdiki = [i]
        for j, cb in enumerate(b, 1):
            simdiki.append(min(onceki[j] + 1, simdiki[j - 1] + 1, onceki[j - 1] + (ca != cb)))
        onceki = simdiki
    return onceki[-1]


def metin_skoru(gercek, okunan):
    """1 - (Levenshtein / uzun olanın uzunluğu). 1 = tam doğru, 0 = hiç benzemiyor."""
    g, o = _duzelt(gercek), _duzelt(okunan)
    if max(len(g), len(o)) == 0:
        return 1.0
    return 1.0 - levenshtein(g, o) / max(len(g), len(o))


def ocr_skoru(img, metin):
    return metin_skoru(metin, ocr_oku(img))


# ---------------------------------------------------------------- eylemler (filtre + parametre)
EYLEM_ADLARI = ["gamma_0.5", "gamma_0.75", "clahe_2", "clahe_4",
                "bilateral_hafif", "bilateral_guclu", "keskin_0.6", "keskin_1.5", "dur"]
DUR = EYLEM_ADLARI.index("dur")


def gamma(img, g):
    # g < 1 aydınlatır (piksel ** g)
    return np.clip(img, 0, 1) ** g


def clahe(img, clip):
    # Sadece Lab'deki L (parlaklık) kanalına. 320x64 görüntüde 8x2 karo = 40x32 piksellik karolar.
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 2)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255


def bilateral(img, d, sigma_renk, sigma_uzay):
    # Kenarları koruyarak düzleştirir (gürültü azaltma)
    return cv2.bilateralFilter(img.astype(np.float32), d=d, sigmaColor=sigma_renk, sigmaSpace=sigma_uzay)


def keskinlestir(img, miktar):
    # Unsharp mask: görüntü + miktar * (görüntü - bulanık hâli)
    bulanik = cv2.GaussianBlur(img, (0, 0), 1.5)
    return np.clip(img + miktar * (img - bulanik), 0, 1)


def eylem_uygula(img, a):
    ad = EYLEM_ADLARI[a]
    if ad == "gamma_0.5":
        return gamma(img, 0.5)
    if ad == "gamma_0.75":
        return gamma(img, 0.75)
    if ad == "clahe_2":
        return clahe(img, 2.0)
    if ad == "clahe_4":
        return clahe(img, 4.0)
    if ad == "bilateral_hafif":
        return bilateral(img, 5, 0.05, 3)
    if ad == "bilateral_guclu":
        return bilateral(img, 9, 0.15, 7)
    if ad == "keskin_0.6":
        return keskinlestir(img, 0.6)
    if ad == "keskin_1.5":
        return keskinlestir(img, 1.5)
    return img  # dur


# ---------------------------------------------------------------- durum
# Sınırlar eğitim verisinden (ilk 100 bozuk görüntü + her birine 3 rastgele filtre
# uygulanmış hâlleri) yaklaşık yüzdelik dilimlere göre seçildi; her aralığa makul sayıda örnek düşüyor.
PARLAKLIK_SINIRLARI = [0.17, 0.26, 0.375]  # -> 4 aralık (%25, %50, %75)
KONTRAST_SINIRLARI = [0.067]               # -> 2 aralık (ortanca)
GURULTU_SINIRLARI = [0.012, 0.09]          # -> 3 aralık (%33, %67)
KESKINLIK_SINIRLARI = [0.035, 0.36]        # -> 3 aralık (%33, %67)
DURUM_SAYISI = 4 * 2 * 3 * 3


def ozellikler(img):
    """Görüntüden 4 sayı: parlaklık, kontrast, gürültü, keskinlik."""
    gri = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    parlaklik = float(gri.mean())
    kontrast = float(gri.std())
    # Gürültü: src/env.py ile aynı (3x3 medyan farkı, parlaklığa göre normalize)
    medyan = cv2.medianBlur((gri * 255).astype(np.uint8), 3).astype(np.float32) / 255
    gurultu = float(np.abs(gri - medyan).mean() / (parlaklik + 0.05))
    # Keskinlik: Laplacian varyansı; parlaklığa göre normalize (karanlık görüntü bulanık görünmesin)
    keskinlik = float(cv2.Laplacian(gri, cv2.CV_32F).var() / (parlaklik + 0.05) ** 2)
    return parlaklik, kontrast, gurultu, keskinlik


def durum_no(img):
    p, k, g, s = ozellikler(img)
    i = int(np.digitize(p, PARLAKLIK_SINIRLARI))
    j = int(np.digitize(k, KONTRAST_SINIRLARI))
    m = int(np.digitize(g, GURULTU_SINIRLARI))
    n = int(np.digitize(s, KESKINLIK_SINIRLARI))
    return ((i * 2 + j) * 3 + m) * 3 + n


# ---------------------------------------------------------------- ortam
class OcrOrtami:
    """
    reset(idx=None) -> durum
    step(a)         -> (yeni_durum, odul, bitti)

    ornekler: veri_uret() çıktısı [(bozuk, temiz, metin, tür), ...]
    Ödül = OCR skoru artışı - adım cezası. "dur": ödül 0, bölüm biter.
    """

    def __init__(self, ornekler, max_adim=5, adim_cezasi=0.02, seed=0):
        self.ornekler = ornekler
        self.max_adim = max_adim
        self.adim_cezasi = adim_cezasi
        self.rng = np.random.default_rng(seed)
        self.eylem_sayisi = len(EYLEM_ADLARI)
        self.durum_sayisi = DURUM_SAYISI
        # Filtreler belirlenimci: aynı görüntüye aynı eylem dizisi hep aynı sonucu verir.
        # OCR yavaş (~0,1 sn) olduğundan skoru (görüntü, eylem dizisi) için bir kez hesaplayıp saklıyoruz.
        self.onbellek = {}

    def _skor(self):
        anahtar = (self.idx, tuple(self.eylemler))
        if anahtar not in self.onbellek:
            self.onbellek[anahtar] = ocr_skoru(self.img, self.metin)
        return self.onbellek[anahtar]

    def reset(self, idx=None):
        if idx is None:
            idx = int(self.rng.integers(len(self.ornekler)))
        self.idx = idx
        self.img, _, self.metin, _ = self.ornekler[idx]
        self.adim = 0
        self.eylemler = []
        self.gecmis = []
        self.skor = self._skor()
        return durum_no(self.img)

    def step(self, a):
        self.adim += 1
        self.gecmis.append(EYLEM_ADLARI[a])
        if a == DUR:
            return durum_no(self.img), 0.0, True
        self.img = eylem_uygula(self.img, a)
        self.eylemler.append(a)
        yeni_skor = self._skor()
        odul = yeni_skor - self.skor - self.adim_cezasi
        self.skor = yeni_skor
        bitti = self.adim >= self.max_adim
        return durum_no(self.img), float(odul), bitti
