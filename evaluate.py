"""
Test (LOL eval15) üzerinde değerlendirme.
Örnek:  python evaluate.py
Çıktı:  results/sonuclar.md (tablo), results/ornekler.png, results/q_tablosu_<ajan>.png
"""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import lol_yukle
from src.env import GoruntuOrtami, EYLEM_ADLARI, DURUM_SAYISI, ssim, psnr
from src.agents import AJANLAR, TabloAjan
from src.baselines import YONTEMLER


def ajan_politikasi(Q):
    """Öğrenilmiş Q-tablosuyla (keşif yok) bir görüntüyü iyileştiren fonksiyon döndürür."""
    def f(dusuk, ref):
        env = GoruntuOrtami([(dusuk, ref)])
        ajan = TabloAjan(DURUM_SAYISI, len(EYLEM_ADLARI))
        ajan.Q = Q
        s, bitti = env.reset(0), False
        while not bitti:
            s, _, bitti = env.step(ajan.eylem_sec(s, ogrenme=False))
        return env.img, env.gecmis
    return f


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--boyut", type=int, default=128)
    p.add_argument("--veri", default="data/LOL")
    a = p.parse_args()

    test, adlar = lol_yukle(a.veri, "eval15", a.boyut, seed=123)
    yontemler = dict(YONTEMLER)
    for ad in AJANLAR:
        if Path(f"results/Q_{ad}.npy").exists():
            yontemler[f"RL: {ad}"] = ajan_politikasi(np.load(f"results/Q_{ad}.npy"))

    satirlar, ciktilar = [], {}
    for yad, f in yontemler.items():
        ps, ss, adim, diziler = [], [], [], Counter()
        ciktilar[yad] = []
        for dusuk, ref in test:
            sonuc, yol = f(dusuk, ref)
            ps.append(psnr(sonuc, ref)); ss.append(ssim(sonuc, ref))
            adim.append(len([e for e in yol if e != "dur"]))
            diziler[" → ".join(yol) or "—"] += 1
            ciktilar[yad].append(sonuc)
        en_sik = diziler.most_common(1)[0]
        satirlar.append((yad, np.mean(ps), np.mean(ss), np.mean(adim), f"{en_sik[0]} ({en_sik[1]}/{len(test)})"))
        print(f"{yad:38s} PSNR {np.mean(ps):6.2f}  SSIM {np.mean(ss):.3f}  ort. adım {np.mean(adim):.1f}")

    Path("results").mkdir(exist_ok=True)
    with open("results/sonuclar.md", "w", encoding="utf-8") as fh:
        fh.write(f"Test: LOL eval15 ({len(test)} görüntü, {a.boyut}x{a.boyut})\n\n")
        fh.write("| Yöntem | PSNR ↑ | SSIM ↑ | Ort. adım |\n|---|---|---|---|\n")
        for y, ps, ss, ad, _ in satirlar:
            fh.write(f"| {y} | {ps:.2f} | {ss:.3f} | {ad:.1f} |\n")
        fh.write("\n**En sık eylem dizisi:**\n\n")
        for y, *_, dz in satirlar:
            fh.write(f"- {y}: {dz}\n")

    # Örnek görseller: ilk 3 test görüntüsü
    secili = [k for k in ["Hiçbir şey (girdi)", "Sabit Gamma + CLAHE", "RL: qlearning"] if k in ciktilar]
    fig, ax = plt.subplots(3, len(secili) + 1, figsize=(3 * (len(secili) + 1), 9))
    for i in range(3):
        for j, k in enumerate(secili):
            ax[i, j].imshow(ciktilar[k][i]); ax[i, j].set_title(k if i == 0 else "", fontsize=9)
        ax[i, -1].imshow(test[i][1]); ax[i, -1].set_title("Referans" if i == 0 else "", fontsize=9)
        for x in ax[i]:
            x.axis("off")
    plt.tight_layout(); plt.savefig("results/ornekler.png", dpi=110)

    # Q-tablosu ısı haritaları
    for ad in AJANLAR:
        yol = Path(f"results/Q_{ad}.npy")
        if yol.exists():
            Q = np.load(yol)
            ziyaret = np.abs(Q).sum(1) > 0
            plt.figure(figsize=(6, 8))
            plt.imshow(Q[ziyaret], aspect="auto", cmap="RdBu", vmin=-np.abs(Q).max(), vmax=np.abs(Q).max())
            plt.colorbar(label="Q değeri")
            plt.xticks(range(len(EYLEM_ADLARI)), EYLEM_ADLARI, rotation=45)
            plt.yticks(range(ziyaret.sum()), np.flatnonzero(ziyaret), fontsize=6)
            plt.ylabel("Durum no (sadece ziyaret edilenler)")
            plt.title(f"Q-tablosu: {ad}"); plt.tight_layout()
            plt.savefig(f"results/q_tablosu_{ad}.png", dpi=110); plt.close()
    print("Kaydedildi: results/sonuclar.md, results/ornekler.png")
