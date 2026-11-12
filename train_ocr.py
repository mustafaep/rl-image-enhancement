"""
İkinci deney (ödül = OCR başarısı) için ajan eğitimi.
Örnek:  python train_ocr.py --ajan hepsi --bolum 3000
        python train_ocr.py --ajan qlearning --bolum 50      (hızlı deneme)
Çıktı:  results_ocr/Q_<ajan>.npy, results_ocr/ogrenme_<ajan>.csv, results_ocr/ogrenme_egrisi.png

Bir OCR çağrısı ~0,1 sn sürdüğü için üç ajan ayrı süreçlerde paralel eğitilir.
"""
import argparse, time, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ocr_env import OcrOrtami, veri_uret
from src.agents import AJANLAR

CIKTI = Path("results_ocr")
ORNEKLER = None   # süreçler arasında paylaşılır (fork)


def egit(ajan_adi, bolum, seed):
    env = OcrOrtami(ORNEKLER, seed=seed)
    ajan = AJANLAR[ajan_adi](env.durum_sayisi, env.eylem_sayisi, seed=seed)
    oduller = []
    t0 = time.time()
    for b in range(bolum):
        oduller.append(ajan.bolum_oyna(env))
        if (b + 1) % 500 == 0:
            print(f"  {ajan_adi:10s} bölüm {b+1:5d}  son 500 ort. ödül {np.mean(oduller[-500:]):+.3f}"
                  f"  epsilon {ajan.epsilon:.2f}  ({time.time()-t0:.0f} sn)", flush=True)
    print(f"  {ajan_adi:10s} bitti: {time.time()-t0:.0f} sn, {len(env.onbellek)} farklı OCR çağrısı", flush=True)
    return ajan_adi, ajan.Q, np.array(oduller)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ajan", default="hepsi", choices=list(AJANLAR) + ["hepsi"])
    p.add_argument("--bolum", type=int, default=3000)
    p.add_argument("--n", type=int, default=400, help="eğitim görüntüsü sayısı")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    CIKTI.mkdir(exist_ok=True)
    t0 = time.time()
    ORNEKLER, elenen = veri_uret(a.n, tohum=0)
    print(f"{len(ORNEKLER)} eğitim görüntüsü üretildi (tohum 0; temiz hâli okunamayan {elenen} örnek elendi, "
          f"{time.time()-t0:.0f} sn).", flush=True)

    adlar = list(AJANLAR) if a.ajan == "hepsi" else [a.ajan]
    with Pool(min(len(adlar), os.cpu_count() or 1)) as havuz:
        sonuclar = havuz.starmap(egit, [(ad, a.bolum, a.seed) for ad in adlar])

    plt.figure(figsize=(8, 4.5))
    pencere = min(100, a.bolum)
    for ad, Q, oduller in sonuclar:
        np.save(CIKTI / f"Q_{ad}.npy", Q)
        np.savetxt(CIKTI / f"ogrenme_{ad}.csv", oduller, delimiter=",")
        plt.plot(np.convolve(oduller, np.ones(pencere) / pencere, mode="valid"), label=ad)
    plt.xlabel("Bölüm (episode)")
    plt.ylabel(f"Toplam ödül ({pencere} bölümlük ortalama)")
    plt.title("Öğrenme eğrisi (ödül = OCR skoru artışı)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(CIKTI / "ogrenme_egrisi.png", dpi=200)
    print(f"Kaydedildi: {CIKTI}/  (toplam {time.time()-t0:.0f} sn)")
