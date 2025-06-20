import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error

# Učitavanje skupa podataka
podaci = pd.read_csv('cars_processed.csv')
print(podaci.info())

# Odabir značajki (feature-a) i ciljne varijable
znacajke = podaci[['km_driven', 'year', 'engine', 'max_power']]
cijene = podaci['selling_price']

# Podjela na skup za treniranje i testiranje
X_train, X_test, y_train, y_test = train_test_split(
    znacajke, cijene, test_size=0.2, random_state=300
)

# Standardizacija podataka
normalizator = StandardScaler()
X_train_norm = normalizator.fit_transform(X_train)
X_test_norm = normalizator.transform(X_test)

# Treniranje linearnog modela
regresija = LinearRegression()
regresija.fit(X_train_norm, y_train)

# Predikcije
train_pred = regresija.predict(X_train_norm)
test_pred = regresija.predict(X_test_norm)

# Evaluacija modela na test skupu
print("R2 score (test):", r2_score(y_test, test_pred))
print("RMSE (test):", np.sqrt(mean_squared_error(y_test, test_pred)))
print("Maksimalna pogreška (test):", max_error(y_test, test_pred))
print("MAE (test):", mean_absolute_error(y_test, test_pred))

# Vizualizacija rezultata
plt.figure(figsize=(13, 10))
sns.regplot(x=test_pred, y=y_test, line_kws={'color': 'green'})
plt.xlabel('Predviđena cijena')
plt.ylabel('Stvarna cijena')
plt.title('Usporedba stvarnih i predviđenih vrijednosti')
plt.grid(True)
plt.show()
