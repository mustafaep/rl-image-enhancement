"""
Ajan eğitimi.
Örnek:  python train.py --ajan qlearning --bolum 3000
        python train.py --ajan hepsi
Çıktı:  results/Q_<ajan>.npy, results/ogrenme_<ajan>.csv, results/ogrenme_egrisi.png
"""
import argparse, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import lol_yukle
from src.env import GoruntuOrtami
from src.agents import AJANLAR


def egit(ajan_adi, ciftler, bolum, seed):
    env = GoruntuOrtami(ciftler, seed=seed)
    ajan = AJANLAR[ajan_adi](env.durum_sayisi, env.eylem_sayisi, seed=seed)
    oduller = []
    t0 = time.time()
    for b in range(bolum):
        oduller.append(ajan.bolum_oyna(env))
        if (b + 1) % 500 == 0:
            print(f"  {ajan_adi:10s} bölüm {b+1:5d}  son 500 ort. ödül {np.mean(oduller[-500:]):+.3f}"
                  f"  epsilon {ajan.epsilon:.2f}  ({time.time()-t0:.0f} sn)")
    return ajan, np.array(oduller)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ajan", default="hepsi", choices=list(AJANLAR) + ["hepsi"])
    p.add_argument("--bolum", type=int, default=3000)
    p.add_argument("--boyut", type=int, default=128)
    p.add_argument("--veri", default="data/LOL")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    Path("results").mkdir(exist_ok=True)
    ciftler, _ = lol_yukle(a.veri, "our485", a.boyut, seed=a.seed)
    print(f"{len(ciftler)} eğitim görüntüsü yüklendi.")

    adlar = list(AJANLAR) if a.ajan == "hepsi" else [a.ajan]
    plt.figure(figsize=(8, 4.5))
    for ad in adlar:
        ajan, oduller = egit(ad, ciftler, a.bolum, a.seed)
        np.save(f"results/Q_{ad}.npy", ajan.Q)
        np.savetxt(f"results/ogrenme_{ad}.csv", oduller, delimiter=",")
        pencere = 100
        plt.plot(np.convolve(oduller, np.ones(pencere) / pencere, mode="valid"), label=ad)
    plt.xlabel("Bölüm (episode)")
    plt.ylabel(f"Toplam ödül ({pencere} bölümlük ortalama)")
    plt.title("Öğrenme eğrisi")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("results/ogrenme_egrisi.png", dpi=130)
    print("Kaydedildi: results/")
