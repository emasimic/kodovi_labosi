
import numpy as np
import matplotlib.pyplot as plt

def generiraj_sahovnicu(dim, r, s):
    tamno = np.zeros((dim, dim), dtype=np.uint8)
    svijetlo = np.ones((dim, dim), dtype=np.uint8) * 255
    red_a = np.hstack([tamno, svijetlo] * (s // 2))
    red_b = np.hstack([svijetlo, tamno] * (s // 2))
    matrica = np.vstack([red_a, red_b] * (r // 2))
    return matrica

dim_kvadrata = 100
r, s = 6, 6

slika = generiraj_sahovnicu(dim_kvadrata, r, s)

plt.imshow(slika, cmap='gray', vmin=0, vmax=255)
plt.show()
