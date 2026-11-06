"""
Tablo tabanlı ajanlar: Q-learning, SARSA, Monte Carlo (her ziyaret, epsilon-açgözlü).

Hepsi aynı Q-tablosunu kullanır: Q[durum, eylem] = "bu durumda bu eylemi
seçersem, bundan sonra toplam ne kadar ödül beklerim?"
Farkları sadece tabloyu NASIL güncelledikleri.
"""
import numpy as np


class TabloAjan:
    def __init__(self, durum_sayisi, eylem_sayisi, alfa=0.1, gama=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_azalma=0.998, seed=0):
        self.Q = np.zeros((durum_sayisi, eylem_sayisi))
        self.alfa, self.gama = alfa, gama                  # öğrenme hızı, gelecek indirimi
        self.epsilon, self.epsilon_min, self.epsilon_azalma = epsilon, epsilon_min, epsilon_azalma
        self.rng = np.random.default_rng(seed)

    def eylem_sec(self, s, ogrenme=True):
        # Keşfet (rastgele) ya da kullan (tablodaki en iyi eylem)
        if ogrenme and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.Q.shape[1]))
        en_iyiler = np.flatnonzero(self.Q[s] == self.Q[s].max())
        return int(self.rng.choice(en_iyiler))           # eşitlikte rastgele seç

    def epsilon_guncelle(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_azalma)


class QLearning(TabloAjan):
    """Hedef: r + gama * max_a' Q(s', a')   (bir sonraki adımda EN İYİSİNİ yapacağımı varsayar)"""

    def bolum_oyna(self, env):
        s, toplam, bitti = env.reset(), 0.0, False
        while not bitti:
            a = self.eylem_sec(s)
            s2, r, bitti = env.step(a)
            hedef = r if bitti else r + self.gama * self.Q[s2].max()
            self.Q[s, a] += self.alfa * (hedef - self.Q[s, a])
            s, toplam = s2, toplam + r
        self.epsilon_guncelle()
        return toplam


class SARSA(TabloAjan):
    """Hedef: r + gama * Q(s', a')   (a' = GERÇEKTEN seçeceğim eylem, keşif dahil)"""

    def bolum_oyna(self, env):
        s, toplam, bitti = env.reset(), 0.0, False
        a = self.eylem_sec(s)
        while not bitti:
            s2, r, bitti = env.step(a)
            a2 = self.eylem_sec(s2)
            hedef = r if bitti else r + self.gama * self.Q[s2, a2]
            self.Q[s, a] += self.alfa * (hedef - self.Q[s, a])
            s, a, toplam = s2, a2, toplam + r
        self.epsilon_guncelle()
        return toplam


class MonteCarlo(TabloAjan):
    """Bölüm bitene kadar bekler, sonra her adımın GERÇEK toplam getirisini (G) hedef alır."""

    def bolum_oyna(self, env):
        s, bitti, yol = env.reset(), False, []
        while not bitti:
            a = self.eylem_sec(s)
            s2, r, bitti = env.step(a)
            yol.append((s, a, r))
            s = s2
        G = 0.0
        for s, a, r in reversed(yol):                    # sondan başa getiriyi topla
            G = r + self.gama * G
            self.Q[s, a] += self.alfa * (G - self.Q[s, a])
        self.epsilon_guncelle()
        return sum(r for _, _, r in yol)


AJANLAR = {"qlearning": QLearning, "sarsa": SARSA, "montecarlo": MonteCarlo}
