"""
LOL veri setini okur: data/LOL/our485/{low,high} (eğitim) ve data/LOL/eval15/{low,high} (test).

Hız için görüntüler küçültülür (varsayılan 128x128). boyut=0 verilirse görüntüler
ORİJİNAL çözünürlükte (LOL: 600x400) ve oranı bozulmadan okunur.
Eylemlerden biri gürültü azaltma olduğu için, istenirse karanlık görüntülerin bir
kısmına yapay gürültü eklenir.
"""
from pathlib import Path
import numpy as np
import cv2


def _oku(yol, boyut):
    img = cv2.cvtColor(cv2.imread(str(yol)), cv2.COLOR_BGR2RGB)
    if boyut:                                   # 0 / None -> orijinal boyut
        img = cv2.resize(img, (boyut, boyut), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255


def lol_yukle(kok="data/LOL", bolum="our485", boyut=128, limit=None,
              gurultu_orani=0.5, gurultu_std=(0.01, 0.04), seed=0):
    kok = Path(kok) / bolum
    dosyalar = sorted((kok / "low").glob("*.png"))
    if not dosyalar:
        raise FileNotFoundError(f"{kok}/low içinde png bulunamadı. data/README.md'ye bak.")
    if limit:
        dosyalar = dosyalar[:limit]
    rng = np.random.default_rng(seed)
    ciftler = []
    for f in dosyalar:
        dusuk, ref = _oku(f, boyut), _oku(kok / "high" / f.name, boyut)
        if rng.random() < gurultu_orani:
            dusuk = np.clip(dusuk + rng.normal(0, rng.uniform(*gurultu_std), dusuk.shape), 0, 1)
        ciftler.append((dusuk.astype(np.float32), ref))
    return ciftler, [f.name for f in dosyalar]
