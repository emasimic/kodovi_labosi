import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage

def generiraj_podatke(broj, tip):
    if tip == 1:
        skup, _ = datasets.make_blobs(n_samples=broj, random_state=365)
    elif tip == 2:
        skup, _ = make_blobs(n_samples=broj, random_state=148)
        skup = np.dot(skup, [[0.6083, -0.6366], [-0.4088, 0.8525]])
    elif tip == 3:
        skup, _ = make_blobs(n_samples=broj, centers=4, cluster_std=[1.0, 2.5, 0.5, 3.0], random_state=148)
    elif tip == 4:
        skup, _ = datasets.make_circles(n_samples=broj, factor=0.5, noise=0.05)
    elif tip == 5:
        skup, _ = datasets.make_moons(n_samples=broj, noise=0.05)
    else:
        skup = []
    return skup

uzorci = 500
tip_skupa = 5
ulazni = generiraj_podatke(uzorci, tip_skupa)

poveznice = linkage(ulazni, method='ward')

plt.figure(figsize=(9, 6))
dendrogram(poveznice)
plt.title('Dendogram - hijerarhijsko grupiranje (ward)')
plt.xlabel('Indeksi uzoraka')
plt.ylabel('Udaljenost')
plt.tight_layout()
plt.show()
