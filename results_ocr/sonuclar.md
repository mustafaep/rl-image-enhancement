Test: sentetik yazı görüntüleri, 100 adet (tohum 123), 320x64. Temiz (bozulmamış) görüntülerin OCR skoru: 1.000 (tavan; temiz hâli okunamayan 16 örnek üretimde elendi).

| Yöntem | Ort. OCR skoru ↑ | Tam okuma ↑ | Ort. adım |
|---|---|---|---|
| Girdi (işlem yok) | 0.772 | %71 | 0.0 |
| Rastgele ajan | 0.600 | %50 | 3.2 |
| Sabit zincir (gamma 0,5 → CLAHE 2,0) | 0.662 | %62 | 2.0 |
| RL: Q-learning | 0.890 | %82 | 0.8 |
| RL: SARSA | 0.827 | %74 | 0.8 |
| RL: Monte Carlo | 0.850 | %77 | 0.2 |
| Açgözlü kâhin (hileli üst sınır) | 0.954 | %88 | 0.3 |

**En sık eylem dizisi:**

- Girdi (işlem yok): dur (100/100)
- Rastgele ajan: dur (9/100)
- Sabit zincir (gamma 0,5 → CLAHE 2,0): gamma_0.5 → clahe_2 → dur (100/100)
- RL: Q-learning: dur (63/100)
- RL: SARSA: dur (71/100)
- RL: Monte Carlo: dur (81/100)
- Açgözlü kâhin (hileli üst sınır): dur (74/100)

**Bozulma türüne göre ortalama OCR skoru:**

| Bozulma | n | Girdi (işlem yok) | Rastgele ajan | Sabit zincir | RL: Q-learning | RL: SARSA | RL: Monte Carlo | Açgözlü kâhin |
|---|---|---|---|---|---|---|---|---|
| yalnız karanlık | 17 | 0.995 | 0.842 | 1.000 | 0.995 | 0.995 | 0.995 | 1.000 |
| karanlık + bulanık | 30 | 0.967 | 0.814 | 0.924 | 0.967 | 0.967 | 0.967 | 1.000 |
| karanlık + gürültülü | 23 | 0.818 | 0.530 | 0.637 | 0.964 | 0.839 | 0.934 | 0.978 |
| hepsi | 30 | 0.416 | 0.303 | 0.228 | 0.696 | 0.584 | 0.586 | 0.862 |
