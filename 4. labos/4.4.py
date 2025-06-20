import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Učitavanje i osnovni pregled podataka
automobili = pd.read_csv('cars_processed.csv')
print(automobili.info())

# Prikaz automobila s najnižom i najvišom cijenom
sortirani = automobili.sort_values("selling_price")
print(sortirani.iloc[[0, -1]])

# Broj vozila proizvedenih 2012. godine
broj_2012 = (automobili["year"] == 2012).sum()
print("Ukupno automobila iz 2012. godine:", broj_2012)

# Prosječna kilometraža po vrsti goriva
prosjek_km = automobili.groupby("fuel")["km_driven"].mean()
print("Prosječna km (Benzin):", prosjek_km.get("Petrol", 0))
print("Prosječna km (Dizel):", prosjek_km.get("Diesel", 0))

# Vizualizacije
sns.pairplot(automobili, hue='fuel')
sns.relplot(data=automobili, x='km_driven', y='selling_price', hue='fuel')

# Uklanjanje nepotrebnih stupaca
automobili.drop(['name', 'mileage'], axis=1, inplace=True)

# Brojačke dijagrame za kategorijske varijable
kategorijske = automobili.select_dtypes(include='object').columns.tolist()
fig, axs = plt.subplots(1, len(kategorijske), figsize=(16, 5))
for kolona, os in zip(kategorijske, axs):
    sns.countplot(x=automobili[kolona], ax=os)

# Kutijasti dijagram prodajnih cijena po vrsti goriva
automobili.boxplot(column='selling_price', by='fuel', grid=False)

# Histogram prodajnih cijena
automobili['selling_price'].hist(grid=False)

plt.tight_layout()
plt.show()
