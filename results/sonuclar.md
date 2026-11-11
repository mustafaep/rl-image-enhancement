Test: LOL eval15 (15 görüntü, 128x128)

| Yöntem | PSNR ↑ | SSIM ↑ | Ort. adım |
|---|---|---|---|
| Hiçbir şey (girdi) | 7.83 | 0.161 | 0.0 |
| Rastgele ajan | 9.84 | 0.335 | 3.1 |
| Histogram eşitleme | 13.91 | 0.506 | 1.0 |
| Sabit Gamma + CLAHE | 14.70 | 0.610 | 2.0 |
| Aç gözlü kâhin (hileli üst sınır) | 18.83 | 0.754 | 3.9 |
| RL: qlearning | 15.91 | 0.713 | 3.8 |
| RL: sarsa | 16.33 | 0.714 | 3.6 |
| RL: montecarlo | 14.96 | 0.682 | 2.9 |

**En sık eylem dizisi:**

- Hiçbir şey (girdi): — (15/15)
- Rastgele ajan: dur (3/15)
- Histogram eşitleme: HE (15/15)
- Sabit Gamma + CLAHE: aydinlat → clahe (15/15)
- Aç gözlü kâhin (hileli üst sınır): aydinlat → aydinlat → keskinlestir (4/15)
- RL: qlearning: gurultu_azalt → aydinlat → aydinlat → dur (4/15)
- RL: sarsa: aydinlat → aydinlat → dur (3/15)
- RL: montecarlo: gurultu_azalt → aydinlat → aydinlat → dur (4/15)
