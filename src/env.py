"""
Görüntü iyileştirme ortamı (environment).

Fikir: Ajan karanlık (ve bazen gürültülü) bir görüntüyle başlar. Her adımda
bir klasik filtre seçer. Görüntü aydınlık referansa ne kadar yaklaşırsa
(SSIM artışı) o kadar ödül alır. "Dur" dediğinde ya da adım sınırı dolunca
bölüm (episode) biter.

Tablo tabanlı RL için durum KESİKLİ olmalı: Görüntünün 3 özelliğini
(parlaklık, kontrast, gürültü) aralıklara bölüp tek bir durum numarasına
çeviriyoruz -> 5 x 4 x 3 = 60 durum.
"""
import numpy as np
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

# ---------------------------------------------------------------- eylemler
EYLEM_ADLARI = ["aydinlat", "karart", "clahe", "gurultu_azalt", "keskinlestir", "dur"]
DUR = EYLEM_ADLARI.index("dur")


def _gamma(img, g):
    # img: 0-1 arası float. g < 1 aydınlatır, g > 1 karartır.
    return np.clip(img, 0, 1) ** g


def _clahe(img):
    # Sadece parlaklık kanalına (Lab'deki L) CLAHE uygula, renkleri bozma.
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255


def _gurultu_azalt(img):
    # Bilateral filtre: kenarları koruyarak düzleştirir.
    return cv2.bilateralFilter(img.astype(np.float32), d=5, sigmaColor=0.1, sigmaSpace=5)


def _keskinlestir(img):
    # Unsharp mask: görüntü + 0.6 * (görüntü - bulanık hâli)
    bulanik = cv2.GaussianBlur(img, (0, 0), 1.5)
    return np.clip(img + 0.6 * (img - bulanik), 0, 1)


def eylem_uygula(img, a):
    if a == 0:
        return _gamma(img, 0.6)
    if a == 1:
        return _gamma(img, 1.25)
    if a == 2:
        return _clahe(img)
    if a == 3:
        return _gurultu_azalt(img)
    if a == 4:
        return _keskinlestir(img)
    return img  # dur


# ---------------------------------------------------------------- durum
PARLAKLIK_SINIRLARI = [0.10, 0.20, 0.35, 0.50]   # -> 5 aralık
KONTRAST_SINIRLARI = [0.05, 0.10, 0.20]          # -> 4 aralık
GURULTU_SINIRLARI = [0.015, 0.035]               # -> 3 aralık
DURUM_SAYISI = 5 * 4 * 3


def ozellikler(img):
    """Görüntüden 3 sayı: ortalama parlaklık, kontrast (std), gürültü tahmini."""
    gri = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    parlaklik = float(gri.mean())
    kontrast = float(gri.std())
    # Gürültü: görüntü ile 3x3 medyan filtreli hâli arasındaki fark.
    # Parlaklığa bölüyoruz, yoksa karanlık görüntü hep "gürültüsüz" görünür.
    medyan = cv2.medianBlur((gri * 255).astype(np.uint8), 3).astype(np.float32) / 255
    gurultu = float(np.abs(gri - medyan).mean() / (parlaklik + 0.05))
    return parlaklik, kontrast, gurultu


def durum_no(img):
    p, k, g = ozellikler(img)
    i = int(np.digitize(p, PARLAKLIK_SINIRLARI))
    j = int(np.digitize(k, KONTRAST_SINIRLARI))
    m = int(np.digitize(g, GURULTU_SINIRLARI))
    return (i * 4 + j) * 3 + m


# ---------------------------------------------------------------- ortam
def ssim(a, b):
    return structural_similarity(a, b, channel_axis=2, data_range=1.0)


def psnr(a, b):
    return peak_signal_noise_ratio(b, a, data_range=1.0)


class GoruntuOrtami:
    """
    reset()  -> durum
    step(a)  -> (yeni_durum, odul, bitti)

    ciftler: [(karanlik, referans), ...] 0-1 arası float32 RGB görüntüler.
    """

    def __init__(self, ciftler, max_adim=5, adim_cezasi=0.01, odul_carpani=10.0, seed=0):
        self.ciftler = ciftler
        self.max_adim = max_adim
        self.adim_cezasi = adim_cezasi
        self.odul_carpani = odul_carpani   # SSIM farkları küçük (0.01 gibi); ölçekliyoruz
        self.rng = np.random.default_rng(seed)
        self.eylem_sayisi = len(EYLEM_ADLARI)
        self.durum_sayisi = DURUM_SAYISI

    def reset(self, idx=None):
        if idx is None:
            idx = int(self.rng.integers(len(self.ciftler)))
        self.img, self.ref = self.ciftler[idx]
        self.adim = 0
        self.skor = ssim(self.img, self.ref)
        self.gecmis = []
        return durum_no(self.img)

    def step(self, a):
        self.adim += 1
        self.gecmis.append(EYLEM_ADLARI[a])
        if a == DUR:
            return durum_no(self.img), 0.0, True
        self.img = eylem_uygula(self.img, a)
        yeni_skor = ssim(self.img, self.ref)
        odul = self.odul_carpani * (yeni_skor - self.skor) - self.adim_cezasi
        self.skor = yeni_skor
        bitti = self.adim >= self.max_adim
        return durum_no(self.img), float(odul), bitti
