import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.datasets import make_blobs

def generiraj_skup(broj, tip):
    if tip == 1:
        stanje = 365
        skup, _ = datasets.make_blobs(n_samples=broj, random_state=stanje)
    elif tip == 2:
        stanje = 148
        skup, _ = make_blobs(n_samples=broj, random_state=stanje)
        matrica = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        skup = np.dot(skup, matrica)
    elif tip == 3:
        stanje = 148
        skup, _ = make_blobs(n_samples=broj, centers=4, cluster_std=[1.0, 2.5, 0.5, 3.0], random_state=stanje)
    elif tip == 4:
        skup, _ = datasets.make_circles(n_samples=broj, factor=.5, noise=.05)
    elif tip == 5:
        skup, _ = datasets.make_moons(n_samples=broj, noise=.05)
    else:
        skup = np.empty((0, 2))
    return skup

def izracunaj_inertnost(podaci, raspon_k):
    return [
        KMeans(n_clusters=k, init='k-means++', random_state=42).fit(podaci).inertia_
        for k in raspon_k
    ]

podaci = generiraj_skup(500, tip=5)
k_list = range(2, 21)
inertnosti = izracunaj_inertnost(podaci, k_list)

plt.figure(figsize=(8, 5))
plt.plot(k_list, inertnosti, marker='o', linestyle='-')
plt.title('Elbow metoda - KMeans')
plt.xlabel('Broj klastera (k)')
plt.ylabel('Vrijednost kriterijske funkcije')
plt.grid(True)
plt.tight_layout()
plt.show()
