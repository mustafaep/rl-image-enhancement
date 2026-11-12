"""
İkinci deney: test kümesinde (sentetik, 100 görüntü, tohum 123) OCR başarısı karşılaştırması.
Örnek:  python evaluate_ocr.py
Çıktı:  results_ocr/sonuclar.md, results_ocr/ornekler.png, results_ocr/q_tablosu_<ajan>.png

Yöntemler: girdi (işlem yok), rastgele ajan, sabit zincir (gamma 0,5 -> CLAHE 2,0),
üç RL ajanı (keşif yok) ve açgözlü kâhin (gerçek metne bakarak her adımda OCR'yi en çok
artıran eylemi seçer; HİLELİ bir üst sınırdır, gerçekte kullanılamaz).
"""
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ocr_env import (veri_uret, eylem_uygula, durum_no, ocr_oku, metin_skoru,
                         EYLEM_ADLARI, DUR, TUR_ADLARI)
from src.agents import AJANLAR

CIKTI = Path("results_ocr")
MAX_ADIM, ADIM_CEZASI = 5, 0.02
SABIT = [EYLEM_ADLARI.index("gamma_0.5"), EYLEM_ADLARI.index("clahe_2")]
AJAN_ADI = {"qlearning": "Q-learning", "sarsa": "SARSA", "montecarlo": "Monte Carlo"}

_okunan = {}   # (görüntü no, eylem dizisi) -> OCR'nin okuduğu metin (aynı şeyi iki kez okutmayalım)


def oku(i, img, dizi):
    anahtar = (i, tuple(dizi))
    if anahtar not in _okunan:
        _okunan[anahtar] = ocr_oku(img)
    return _okunan[anahtar]


def calistir(i, ornek, secici):
    """secici(i, img, dizi, metin) -> eylem. Sonuç görüntüsü, eylem adları, okunan metin, skor."""
    img, _, metin, _ = ornek
    dizi, yol = [], []
    for _ in range(MAX_ADIM):
        a = secici(i, img, dizi, metin)
        yol.append(EYLEM_ADLARI[a])
        if a == DUR:
            break
        img = eylem_uygula(img, a)
        dizi.append(a)
    okunan = oku(i, img, dizi)
    return img, yol, okunan, metin_skoru(metin, okunan)


def girdi(i, img, dizi, metin):
    return DUR


def rastgele_secici():
    rng = np.random.default_rng(0)
    return lambda i, img, dizi, metin: int(rng.integers(len(EYLEM_ADLARI)))


def sabit(i, img, dizi, metin):
    return SABIT[len(dizi)] if len(dizi) < len(SABIT) else DUR


def ajan_secici(Q):
    def f(i, img, dizi, metin):
        s = durum_no(img)
        return int(Q[s].argmax()) if Q[s].max() > 0 else DUR    # hiçbir eylem işe yaramıyorsa dur
    return f


def kahin(i, img, dizi, metin):
    simdi = metin_skoru(metin, oku(i, img, dizi))
    kazanc = [metin_skoru(metin, oku(i, eylem_uygula(img, a), dizi + [a])) - simdi
              for a in range(len(EYLEM_ADLARI)) if a != DUR]
    en_iyi = int(np.argmax(kazanc))
    return en_iyi if kazanc[en_iyi] - ADIM_CEZASI > 0 else DUR


if __name__ == "__main__":
    test, elenen = veri_uret(100, tohum=123)
    temiz_skor = np.mean([metin_skoru(m, ocr_oku(t)) for _, t, m, _ in test])
    print(f"{len(test)} test görüntüsü (tohum 123, {elenen} elendi). Temiz görüntülerin OCR skoru: {temiz_skor:.3f}")

    yontemler = {"Girdi (işlem yok)": girdi, "Rastgele ajan": rastgele_secici(),
                 "Sabit zincir (gamma 0,5 → CLAHE 2,0)": sabit}
    for ad in AJANLAR:
        if (CIKTI / f"Q_{ad}.npy").exists():
            yontemler[f"RL: {AJAN_ADI[ad]}"] = ajan_secici(np.load(CIKTI / f"Q_{ad}.npy"))
    yontemler["Açgözlü kâhin (hileli üst sınır)"] = kahin

    sonuc = {}
    for yad, f in yontemler.items():
        sonuc[yad] = [calistir(i, o, f) for i, o in enumerate(test)]
        sk = np.array([r[3] for r in sonuc[yad]])
        print(f"{yad:40s} OCR {sk.mean():.3f}  tam okuma %{100 * (sk == 1).mean():.0f}", flush=True)

    turler = np.array([o[3] for o in test])
    with open(CIKTI / "sonuclar.md", "w", encoding="utf-8") as fh:
        fh.write(f"Test: sentetik yazı görüntüleri, {len(test)} adet (tohum 123), 320x64. "
                 f"Temiz (bozulmamış) görüntülerin OCR skoru: {temiz_skor:.3f} (tavan; temiz hâli okunamayan "
                 f"{elenen} örnek üretimde elendi).\n\n")
        fh.write("| Yöntem | Ort. OCR skoru ↑ | Tam okuma ↑ | Ort. adım |\n|---|---|---|---|\n")
        for yad, rs in sonuc.items():
            sk = np.array([r[3] for r in rs])
            adim = np.mean([len([e for e in r[1] if e != "dur"]) for r in rs])
            fh.write(f"| {yad} | {sk.mean():.3f} | %{100 * (sk == 1).mean():.0f} | {adim:.1f} |\n")
        fh.write("\n**En sık eylem dizisi:**\n\n")
        for yad, rs in sonuc.items():
            dz = Counter(" → ".join(r[1]) or "—" for r in rs).most_common(1)[0]
            fh.write(f"- {yad}: {dz[0]} ({dz[1]}/{len(test)})\n")
        fh.write("\n**Bozulma türüne göre ortalama OCR skoru:**\n\n")
        kisa = list(yontemler)
        fh.write("| Bozulma | n | " + " | ".join(k.replace("Sabit zincir (gamma 0,5 → CLAHE 2,0)", "Sabit zincir")
                                             .replace(" (hileli üst sınır)", "") for k in kisa) + " |\n")
        fh.write("|---|---|" + "---|" * len(kisa) + "\n")
        for t, tad in enumerate(TUR_ADLARI):
            m = turler == t
            fh.write(f"| {tad} | {m.sum()} | " + " | ".join(
                f"{np.mean([r[3] for r, mm in zip(sonuc[k], m) if mm]):.3f}" for k in kisa) + " |\n")

    # Örnekler: ajanın sabit zinciri geçtiği, geride kaldığı ve eşit kaldığı durumlardan ilk ikişer tane
    s_ad, q_ad = "Sabit zincir (gamma 0,5 → CLAHE 2,0)", "RL: Q-learning"
    if q_ad in sonuc:
        fark = [sonuc[q_ad][i][3] - sonuc[s_ad][i][3] for i in range(len(test))]
        gruplar = [[i for i, d in enumerate(fark) if d > 0], [i for i, d in enumerate(fark) if d < 0],
                   [i for i, d in enumerate(fark) if d == 0]]
        secili = [i for g in gruplar for i in g[:2]]
        secili += [i for i in range(len(test)) if i not in secili][:6 - len(secili)]
        fig, ax = plt.subplots(6, 4, figsize=(16, 6 * 1.25))
        for r, i in enumerate(secili):
            paneller = [(test[i][0], "Girdi", sonuc["Girdi (işlem yok)"][i]),
                        (sonuc[s_ad][i][0], "Sabit zincir", sonuc[s_ad][i]),
                        (sonuc[q_ad][i][0], "Q-learning: " + " → ".join(sonuc[q_ad][i][1]), sonuc[q_ad][i])]
            for c, (img, b, rs) in enumerate(paneller):
                ax[r, c].imshow(np.clip(img, 0, 1), interpolation="none")
                ax[r, c].set_title(f"{b}\n\"{rs[2].strip()}\" · skor {rs[3]:.2f}", fontsize=8)
            ax[r, 3].imshow(test[i][1], interpolation="none")
            ax[r, 3].set_title(f"Temiz (gerçek: \"{test[i][2]}\")\n{TUR_ADLARI[test[i][3]]}", fontsize=8)
            for x in ax[r]:
                x.axis("off")
        plt.tight_layout(); plt.savefig(CIKTI / "ornekler.png", dpi=150); plt.close()

    # Q-tablosu ısı haritaları (evaluate.py ile aynı biçim)
    for ad in AJANLAR:
        yol = CIKTI / f"Q_{ad}.npy"
        if yol.exists():
            Q = np.load(yol)
            ziyaret = np.abs(Q).sum(1) > 0
            plt.figure(figsize=(7, 9))
            plt.imshow(Q[ziyaret], aspect="auto", cmap="RdBu", vmin=-np.abs(Q).max(), vmax=np.abs(Q).max())
            plt.colorbar(label="Q değeri")
            plt.xticks(range(len(EYLEM_ADLARI)), EYLEM_ADLARI, rotation=45)
            plt.yticks(range(ziyaret.sum()), np.flatnonzero(ziyaret), fontsize=6)
            plt.ylabel("Durum no (sadece ziyaret edilenler)")
            plt.title(f"Q-tablosu (OCR ödülü): {ad}"); plt.tight_layout()
            plt.savefig(CIKTI / f"q_tablosu_{ad}.png", dpi=200); plt.close()
    print(f"Kaydedildi: {CIKTI}/sonuclar.md, ornekler.png, q_tablosu_*.png")
