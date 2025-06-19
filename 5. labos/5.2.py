import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

podaci = pd.read_csv('occupancy_processed.csv')
ulazi = podaci.loc[:, ['S3_Temp', 'S5_CO2']].values
oznaka = podaci['Room_Occupancy_Count'].astype(int)

ulaz_train, ulaz_test, oznaka_train, oznaka_test = train_test_split(
    ulazi, oznaka, test_size=0.2, stratify=oznaka, shuffle=True, random_state=123
)

skaliraj = StandardScaler()
ulaz_train_skaliran = skaliraj.fit_transform(ulaz_train)
ulaz_test_skaliran = skaliraj.transform(ulaz_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(ulaz_train_skaliran, oznaka_train)

predikcija = knn.predict(ulaz_test_skaliran)

matrica = confusion_matrix(oznaka_test, predikcija)
ConfusionMatrixDisplay(matrica, display_labels=["Slobodna", "Zauzeta"]).plot(cmap="Oranges")
plt.title("Matrica zabune KNN klasifikatora")
plt.grid(False)
plt.show()

print("Rezultati klasifikacije:\n")
print(classification_report(oznaka_test, predikcija, target_names=["Slobodna", "Zauzeta"]))
