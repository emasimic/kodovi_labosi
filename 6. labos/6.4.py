from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

slika = mpimg.imread('example_grayscale.png')
if slika.ndim == 3:
    slika = np.mean(slika, axis=2)

pikseli = slika.reshape(-1, 1)

k_broj = 10
model = cluster.KMeans(n_clusters=k_broj, n_init=10)
model.fit(pikseli)

centri = model.cluster_centers_.squeeze()
oznake = model.labels_
komprimirano = np.choose(oznake, centri).reshape(slika.shape)

plt.figure()
plt.imshow(slika, cmap='gray')
plt.title('Originalna slika')
plt.axis('off')

plt.figure()
plt.imshow(komprimirano, cmap='gray')
plt.title(f'Kvantizirana slika (k = {k_broj})')
plt.axis('off')
plt.show()
