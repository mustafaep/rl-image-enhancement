"""
İkinci deney: eğitilmiş ajanı TEK bir yazı görüntüsünde adım adım çalıştırır.

Örnek:  python demo_ocr.py --tohum 7                         (sentetik bir test görüntüsü üretir)
        python demo_ocr.py dosya.png --metin "KITAP 4827"    (kendi görüntün; --metin yoksa sadece okunan yazılır)
        python demo_ocr.py --tohum 7 --ajan sarsa

Çıktı:  results_ocr/demo/<ad>_<ajan>.png  (girdi · sabit zincir · ajan · varsa temiz hâli)

Ajan gerçek metni bilmez: görüntüden durumu hesaplar, Q-tablosundan eylemi okur.
Gerçek metin verilirse sadece OCR skorunu raporlamak için kullanılır.
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ocr_env import (EYLEM_ADLARI, DUR, eylem_uygula, durum_no, ozellikler, ocr_oku, metin_skoru,
                         veri_uret, TUR_ADLARI)


def skor_yazi(okunan, metin):
    return f"\"{okunan.strip()}\"" + (f" (skor {metin_skoru(metin, okunan):.2f})" if metin else "")


def calistir(img, Q, metin=None, max_adim=5):
    """Açgözlü politika (keşif yok). Hiçbir eylemin değeri 0'ın üstünde değilse durur."""
    yol = []
    print(f"  başlangıç: OCR {skor_yazi(ocr_oku(img), metin)}")
    for adim in range(1, max_adim + 1):
        s = durum_no(img)
        p, k, g, ks = ozellikler(img)
        a = int(Q[s].argmax()) if Q[s].max() > 0 else DUR
        satir = (f"  {adim}. adım | durum {s:2d} (parlaklık {p:.2f}, kontrast {k:.2f}, gürültü {g:.3f}, "
                 f"keskinlik {ks:.3f}) -> {EYLEM_ADLARI[a]}")
        if not Q[s].any():
            satir += "   (bu durum eğitimde hiç görülmedi, ajan duruyor)"
        yol.append(EYLEM_ADLARI[a])
        if a == DUR:
            print(satir)
            break
        img = eylem_uygula(img, a)
        print(satir + f" | OCR {skor_yazi(ocr_oku(img), metin)}")
    return img, yol


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Eğitilmiş OCR ajanını tek bir görüntüde çalıştır.")
    p.add_argument("goruntu", nargs="?", help="görüntü dosyası (tek satır yazı)")
    p.add_argument("--metin", help="görüntüdeki gerçek metin (isteğe bağlı, sadece skor için)")
    p.add_argument("--tohum", type=int, help="bu tohumla sentetik bir test görüntüsü üret")
    p.add_argument("--ajan", default="qlearning", choices=["qlearning", "sarsa", "montecarlo"])
    a = p.parse_args()

    q_yolu = Path(f"results_ocr/Q_{a.ajan}.npy")
    if not q_yolu.exists():
        raise SystemExit(f"{q_yolu} yok. Önce eğit: python train_ocr.py --ajan {a.ajan}")
    Q = np.load(q_yolu)

    temiz = None
    if a.goruntu:
        img = cv2.imread(a.goruntu)
        if img is None:
            raise SystemExit(f"Görüntü okunamadı: {a.goruntu}")
        girdi = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        ad, metin = Path(a.goruntu).stem, a.metin
    elif a.tohum is not None:
        (girdi, temiz, metin, tur), = veri_uret(1, a.tohum)[0]
        ad = f"tohum{a.tohum}"
        print(f"Sentetik görüntü (tohum {a.tohum}): gerçek metin \"{metin}\", bozulma: {TUR_ADLARI[tur]}")
    else:
        p.error("bir görüntü dosyası ya da --tohum ver")

    print(f"Görüntü: {ad} ({girdi.shape[1]}x{girdi.shape[0]})   Ajan: {a.ajan}")
    sonuc, yol = calistir(girdi, Q, metin)
    sabit = eylem_uygula(eylem_uygula(girdi, EYLEM_ADLARI.index("gamma_0.5")), EYLEM_ADLARI.index("clahe_2"))
    print("Ajanın kararı: " + " → ".join(yol))
    print(f"OCR: girdi {skor_yazi(ocr_oku(girdi), metin)} | sabit zincir {skor_yazi(ocr_oku(sabit), metin)}"
          f" | ajan {skor_yazi(ocr_oku(sonuc), metin)}")

    paneller = [(girdi, "Girdi"), (sabit, "Sabit: gamma 0,5 → CLAHE 2,0"),
                (sonuc, f"Ajan ({a.ajan}): " + " → ".join(yol))]
    if temiz is not None:
        paneller.append((temiz, "Temiz"))
    fig, ax = plt.subplots(len(paneller), 1, figsize=(6, 1.6 * len(paneller)))
    for x, (im, baslik) in zip(ax, paneller):
        x.imshow(np.clip(im, 0, 1), interpolation="none")
        x.set_title(f"{baslik}\nOCR {skor_yazi(ocr_oku(im), metin)}", fontsize=8); x.axis("off")
    cikti = Path("results_ocr/demo"); cikti.mkdir(parents=True, exist_ok=True)
    yol_png = cikti / f"{ad}_{a.ajan}.png"
    plt.tight_layout(); plt.savefig(yol_png, dpi=150); plt.close()
    print(f"Kaydedildi: {yol_png}")
