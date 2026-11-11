Test: LOL eval15 (15 görüntü, 600x400)

| Yöntem | PSNR ↑ | SSIM ↑ | Ort. adım |
|---|---|---|---|
| Hiçbir şey (girdi) | 7.77 | 0.173 | 0.0 |
| Rastgele ajan | 9.47 | 0.276 | 3.1 |
| Histogram eşitleme | 13.98 | 0.334 | 1.0 |
| Sabit Gamma + CLAHE | 13.69 | 0.440 | 2.0 |
| Aç gözlü kâhin (hileli üst sınır) | 17.46 | 0.692 | 4.5 |
| RL: qlearning | 15.55 | 0.575 | 3.7 |
| RL: sarsa | 15.64 | 0.558 | 3.5 |
| RL: montecarlo | 14.30 | 0.526 | 2.8 |

**En sık eylem dizisi:**

- Hiçbir şey (girdi): — (15/15)
- Rastgele ajan: dur (3/15)
- Histogram eşitleme: HE (15/15)
- Sabit Gamma + CLAHE: aydinlat → clahe (15/15)
- Aç gözlü kâhin (hileli üst sınır): aydinlat → aydinlat → gurultu_azalt → keskinlestir (3/15)
- RL: qlearning: aydinlat → aydinlat → dur (4/15)
- RL: sarsa: aydinlat → aydinlat → dur (5/15)
- RL: montecarlo: aydinlat → aydinlat → dur (6/15)
