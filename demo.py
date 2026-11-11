"""
Eğitilmiş ajanı TEK bir görüntüde çalıştırır ve her adımı ekrana yazar.

Örnek:  python demo.py --test 22                      (LOL test görüntüsü, referansıyla)
        python demo.py fotograf.jpg                   (herhangi bir fotoğraf, referans yok)
        python demo.py karanlik.png --referans aydinlik.png
        python demo.py --test 22 --ajan sarsa

Çıktı:  results/demo/<ad>.png  (girdi · Sabit Gamma + CLAHE · ajan · varsa referans)

Ajan referansa ihtiyaç duymaz: görüntüden durumu hesaplar, Q-tablosundan eylemi okur.
Referans verilirse sadece SSIM'i raporlamak için kullanılır.
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.env import EYLEM_ADLARI, DUR, eylem_uygula, durum_no, ozellikler, ssim


def oku(yol, maks):
    img = cv2.imread(str(yol))
    if img is None:
        raise SystemExit(f"Görüntü okunamadı: {yol}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if maks and max(h, w) > maks:                      # çok büyük fotoğrafı küçült (hız için)
        o = maks / max(h, w)
        img = cv2.resize(img, (round(w * o), round(h * o)), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255


def calistir(img, Q, ref=None, max_adim=5):
    """Açgözlü politika (keşif yok). Hiçbir eylemin değeri 0'ın üstünde değilse durur."""
    yol = []
    for adim in range(1, max_adim + 1):
        s = durum_no(img)
        p, k, g = ozellikler(img)
        gorulmedi = not Q[s].any()
        a = int(Q[s].argmax()) if Q[s].max() > 0 else DUR
        satir = f"  {adim}. adım | durum {s:2d} (parlaklık {p:.2f}, kontrast {k:.2f}, gürültü {g:.3f})"
        satir += f" | Q = [{', '.join(f'{q:+.2f}' for q in Q[s])}] -> {EYLEM_ADLARI[a]}"
        if gorulmedi:
            satir += "   (bu durum eğitimde hiç görülmedi, ajan duruyor)"
        yol.append(EYLEM_ADLARI[a])
        if a == DUR:
            print(satir)
            break
        img = eylem_uygula(img, a)
        if ref is not None:
            satir += f" | SSIM {ssim(img, ref):.3f}"
        print(satir)
    return img, yol


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Eğitilmiş ajanı tek bir görüntüde çalıştır.")
    p.add_argument("goruntu", nargs="?", help="görüntü dosyası")
    p.add_argument("--referans", help="aydınlık referans (isteğe bağlı, sadece SSIM için)")
    p.add_argument("--test", help="LOL test görüntüsünün adı, ör. 22 (data/LOL/eval15)")
    p.add_argument("--ajan", default="qlearning", choices=["qlearning", "sarsa", "montecarlo"])
    p.add_argument("--maks", type=int, default=600, help="uzun kenar bundan büyükse küçült (0 = dokunma)")
    a = p.parse_args()

    if a.test:
        a.goruntu = f"data/LOL/eval15/low/{a.test}.png"
        a.referans = f"data/LOL/eval15/high/{a.test}.png"
    if not a.goruntu:
        p.error("bir görüntü dosyası ya da --test ver")
    q_yolu = Path(f"results/Q_{a.ajan}.npy")
    if not q_yolu.exists():
        raise SystemExit(f"{q_yolu} yok. Önce eğit: python train.py --ajan {a.ajan}")

    Q = np.load(q_yolu)
    girdi = oku(a.goruntu, a.maks)
    ref = oku(a.referans, a.maks) if a.referans else None
    if ref is not None and ref.shape != girdi.shape:
        raise SystemExit("Referans ile girdi aynı boyutta olmalı.")

    print(f"Görüntü: {a.goruntu}  ({girdi.shape[1]}x{girdi.shape[0]})   Ajan: {a.ajan}")
    print("  Q sırası: " + ", ".join(EYLEM_ADLARI))
    if ref is not None:
        print(f"  başlangıç SSIM {ssim(girdi, ref):.3f}")
    sonuc, yol = calistir(girdi, Q, ref)
    sabit = eylem_uygula(eylem_uygula(girdi, 0), 2)
    print("Ajanın kararı: " + " → ".join(yol))
    if ref is not None:
        print(f"SSIM: girdi {ssim(girdi, ref):.3f} | Sabit Gamma + CLAHE {ssim(sabit, ref):.3f} | ajan {ssim(sonuc, ref):.3f}")

    paneller = [(girdi, "Girdi"), (sabit, "Sabit Gamma + CLAHE"), (sonuc, f"Ajan ({a.ajan}): " + " → ".join(yol))]
    if ref is not None:
        paneller.append((ref, "Referans"))
    h, w = girdi.shape[:2]
    fig, ax = plt.subplots(1, len(paneller), figsize=(5 * len(paneller), 5 * h / w + 0.6))
    for x, (im, baslik) in zip(ax, paneller):
        x.imshow(np.clip(im, 0, 1), interpolation="none"); x.set_title(baslik, fontsize=9); x.axis("off")
    cikti = Path("results/demo"); cikti.mkdir(parents=True, exist_ok=True)
    yol_png = cikti / f"{Path(a.goruntu).stem}_{a.ajan}.png"
    plt.tight_layout(); plt.savefig(yol_png, dpi=150); plt.close()
    print(f"Kaydedildi: {yol_png}")
