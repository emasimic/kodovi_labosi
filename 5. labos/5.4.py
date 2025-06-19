from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

podaci = pd.read_csv("occupancy_processed.csv")

ulazi = podaci.loc[:, ['S3_Temp', 'S5_CO2']].to_numpy()
oznaka = podaci['Room_Occupancy_Count'].to_numpy()

ulaz_train, ulaz_test, oznaka_train, oznaka_test = train_test_split(
    ulazi, oznaka, test_size=0.2, stratify=oznaka, random_state=42
)

skaliraj = StandardScaler()
skaliraj.fit(ulaz_train)
ulaz_train = skaliraj.transform(ulaz_train)
ulaz_test = skaliraj.transform(ulaz_test)

model = LogisticRegression()
model.fit(ulaz_train, oznaka_train)

predikcija = model.predict(ulaz_test)

matrica = confusion_matrix(oznaka_test, predikcija)
ConfusionMatrixDisplay(matrica, display_labels=["Slobodna", "Zauzeta"]).plot(cmap="Oranges")
plt.title("Matrica zabune - Logistička regresija")
plt.grid(False)
plt.show()

print("Točnost modela:", accuracy_score(oznaka_test, predikcija))
print("Rezultati klasifikacije:\n")
print(classification_report(oznaka_test, predikcija, target_names=['Slobodna', 'Zauzeta']))
