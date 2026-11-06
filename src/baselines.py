"""
Karşılaştırma yöntemleri. Hepsi aynı arayüz: f(dusuk, ref) -> (sonuc, eylem_listesi)
(ref sadece "aç gözlü kâhin"de kullanılır; o yöntem referansı gördüğü için HİLELİ bir üst sınırdır.)
"""
import numpy as np
import cv2
from .env import eylem_uygula, EYLEM_ADLARI, DUR, ssim


def hicbir_sey(dusuk, ref):
    return dusuk, []


def sabit_gamma_clahe(dusuk, ref):
    img = eylem_uygula(eylem_uygula(dusuk, 0), 2)
    return img, ["aydinlat", "clahe"]


def histogram_esitleme(dusuk, ref):
    u8 = (dusuk * 255).astype(np.uint8)
    ycc = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
    ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255, ["HE"]


_RNG = np.random.default_rng(0)   # tek üreteç: her görüntüde farklı rastgele seçim


def rastgele(dusuk, ref, max_adim=5):
    rng = _RNG
    img, yol = dusuk, []
    for _ in range(max_adim):
        a = int(rng.integers(len(EYLEM_ADLARI)))
        yol.append(EYLEM_ADLARI[a])
        if a == DUR:
            break
        img = eylem_uygula(img, a)
    return img, yol


def acgozlu_kahin(dusuk, ref, max_adim=5):
    """Her adımda referansa bakıp SSIM'i en çok artıran eylemi seçer. Gerçekte yapılamaz (referans yok)."""
    img, yol, skor = dusuk, [], ssim(dusuk, ref)
    for _ in range(max_adim):
        adaylar = [(ssim(eylem_uygula(img, a), ref), a) for a in range(len(EYLEM_ADLARI)) if a != DUR]
        en_iyi, a = max(adaylar)
        if en_iyi <= skor:
            break
        img, skor = eylem_uygula(img, a), en_iyi
        yol.append(EYLEM_ADLARI[a])
    return img, yol


YONTEMLER = {
    "Hiçbir şey (girdi)": hicbir_sey,
    "Rastgele ajan": rastgele,
    "Histogram eşitleme": histogram_esitleme,
    "Sabit Gamma + CLAHE": sabit_gamma_clahe,
    "Aç gözlü kâhin (hileli üst sınır)": acgozlu_kahin,
}
