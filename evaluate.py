"""
Test (LOL eval15) üzerinde değerlendirme.
Örnek:  python evaluate.py            (128x128, eğitimle aynı boyut)
        python evaluate.py --tam      (orijinal 600x400 çözünürlük)
Çıktı:  results/sonuclar[_tam].md (tablo), results/ornekler[_tam].png, results/q_tablosu_<ajan>.png
        --tam ile ayrıca: results/tam/<görüntü>.png (her test görüntüsü için piksel piksel karşılaştırma)

Ajan 128x128'de eğitilir; durum her zaman 128x128'lik bir kopyadan hesaplandığı için
aynı Q-tablosu tam çözünürlükte de kullanılabilir. Filtreler ise tam çözünürlüklü
görüntünün kendisine uygulanır.
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


def piksel_tam_karsilastirma(gorseller, basliklar, yol, baslik_h=48, bosluk=14):
    """Görüntüleri HİÇ ÖLÇEKLEMEDEN yan yana kaydeder: 1 görüntü pikseli = 1 dosya pikseli."""
    h, w = gorseller[0].shape[:2]
    n = len(gorseller)
    W, H = n * w + (n + 1) * bosluk, h + baslik_h + bosluk
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
    for k, (img, b) in enumerate(zip(gorseller, basliklar)):
        x = bosluk + k * (w + bosluk)
        ax = fig.add_axes([x / W, bosluk / H, w / W, h / H])
        ax.imshow(np.clip(img, 0, 1), interpolation="none")
        ax.axis("off")
        fig.text((x + w / 2) / W, (bosluk + h + baslik_h / 2) / H, b, ha="center", va="center", fontsize=14)
    fig.savefig(yol, dpi=100)
    plt.close(fig)


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
    p.add_argument("--tam", action="store_true", help="orijinal çözünürlükte değerlendir (600x400)")
    p.add_argument("--veri", default="data/LOL")
    a = p.parse_args()
    if a.tam:
        a.boyut = 0
    ek = "_tam" if a.tam else ""

    test, adlar = lol_yukle(a.veri, "eval15", a.boyut, seed=123)
    h, w = test[0][0].shape[:2]
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
    with open(f"results/sonuclar{ek}.md", "w", encoding="utf-8") as fh:
        fh.write(f"Test: LOL eval15 ({len(test)} görüntü, {w}x{h})\n\n")
        fh.write("| Yöntem | PSNR ↑ | SSIM ↑ | Ort. adım |\n|---|---|---|---|\n")
        for y, ps, ss, ad, _ in satirlar:
            fh.write(f"| {y} | {ps:.2f} | {ss:.3f} | {ad:.1f} |\n")
        fh.write("\n**En sık eylem dizisi:**\n\n")
        for y, *_, dz in satirlar:
            fh.write(f"- {y}: {dz}\n")

    # Örnek görseller: ilk 3 test görüntüsü
    secili = [k for k in ["Hiçbir şey (girdi)", "Sabit Gamma + CLAHE", "RL: qlearning"] if k in ciktilar]
    oran = h / w
    fig, ax = plt.subplots(3, len(secili) + 1, figsize=(4 * (len(secili) + 1), 12 * oran + 0.6))
    for i in range(3):
        for j, k in enumerate(secili):
            ax[i, j].imshow(np.clip(ciktilar[k][i], 0, 1), interpolation="none")
            ax[i, j].set_title(k if i == 0 else "", fontsize=10)
        ax[i, -1].imshow(test[i][1], interpolation="none"); ax[i, -1].set_title("Referans" if i == 0 else "", fontsize=10)
        for x in ax[i]:
            x.axis("off")
    plt.tight_layout(); plt.savefig(f"results/ornekler{ek}.png", dpi=200); plt.close()

    # Tam çözünürlükte: her test görüntüsü için piksel piksel karşılaştırma
    if a.tam:
        Path("results/tam").mkdir(exist_ok=True)
        panel = [k for k in ["Hiçbir şey (girdi)", "Sabit Gamma + CLAHE", "RL: qlearning", "RL: sarsa"] if k in ciktilar]
        for i, (ad_dosya, (dusuk, ref)) in enumerate(zip(adlar, test)):
            gorseller = [ciktilar[k][i] for k in panel] + [ref]
            basliklar = [f"{k} · SSIM {ssim(ciktilar[k][i], ref):.3f}" for k in panel] + ["Referans (aydınlık)"]
            piksel_tam_karsilastirma(gorseller, basliklar, f"results/tam/{Path(ad_dosya).stem}.png")
        print(f"Kaydedildi: results/tam/ ({len(test)} karşılaştırma, {w}x{h} piksel, ölçeklenmeden)")

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
            plt.savefig(f"results/q_tablosu_{ad}.png", dpi=200); plt.close()
    print(f"Kaydedildi: results/sonuclar{ek}.md, results/ornekler{ek}.png")
