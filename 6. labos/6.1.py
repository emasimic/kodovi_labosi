import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

def generiraj_podatke(broj, opcija):
    if opcija == 1:
        stanje = 365
        X, _ = datasets.make_blobs(n_samples=broj, random_state=stanje)
    elif opcija == 2:
        stanje = 148
        X, _ = datasets.make_blobs(n_samples=broj, random_state=stanje)
        transformacija = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformacija)
    elif opcija == 3:
        stanje = 148
        X, _ = datasets.make_blobs(n_samples=broj, centers=4,
                                   cluster_std=[1.0, 2.5, 0.5, 3.0], random_state=stanje)
    elif opcija == 4:
        X, _ = datasets.make_circles(n_samples=broj, factor=0.5, noise=0.05)
    elif opcija == 5:
        X, _ = datasets.make_moons(n_samples=broj, noise=0.05)
    else:
        X = []
    return X

uzorci = 500
izbor = 5

skup = generiraj_podatke(uzorci, izbor)

plt.figure(figsize=(6, 5))
plt.scatter(skup[:, 0], skup[:, 1], s=15, color='gray')
plt.title("Ulazni podaci")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.tight_layout()
plt.show()

model = KMeans(n_clusters=3, n_init=10, random_state=42)
grupe = model.fit_predict(skup)

plt.figure(figsize=(6, 5))
plt.scatter(skup[:, 0], skup[:, 1], c=grupe, s=15, cmap='plasma')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
            s=180, c='white', edgecolors='black', marker='X')
plt.title("Rezultat KMeans algoritma")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.tight_layout()
plt.show()
