import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

putanja = 'occupancy_processed.csv'
podaci = pd.read_csv(putanja)
print(f"Ukupno primjera: {len(podaci)}")

znacajke = ['S3_Temp', 'S5_CO2']
oznaka = 'Room_Occupancy_Count'
klase = {0: 'Slobodna', 1: 'Zauzeta'}

X = podaci[znacajke].values
y = podaci[oznaka].values

plt.figure(figsize=(8, 6))
for vrijednost in np.unique(y):
    plt.scatter(
        X[y == vrijednost, 0],
        X[y == vrijednost, 1],
        label=klase[vrijednost],
        alpha=0.7
    )

plt.xlabel('Temperatura (S3_Temp)')
plt.ylabel('CO2 razina (S5_CO2)')
plt.title('Raspršeni dijagram zauzetosti prostorije')
plt.legend()
plt.show()
