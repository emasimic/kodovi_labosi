import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.io import imread, imsave

ulazna_slika = imread("example.png")
visina, sirina, kanali = ulazna_slika.shape

pikseli = ulazna_slika.reshape(-1, 3)

k = 12
model = KMeans(n_clusters=k, random_state=42)
model.fit(pikseli)

kvantizirani_pikseli = model.cluster_centers_[model.labels_]
izlazna_slika = kvantizirani_pikseli.reshape(visina, sirina, kanali).astype(np.uint8)

imsave("quantized_example.png", izlazna_slika)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(ulazna_slika)
plt.title("Originalna slika")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(izlazna_slika)
plt.title("Kvantizirana slika")
plt.axis('off')

plt.tight_layout()
plt.show()
