import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

podaci = pd.read_csv('occupancy_processed.csv')

ulazi = podaci.loc[:, ['S3_Temp', 'S5_CO2']].to_numpy()
oznaka = podaci['Room_Occupancy_Count'].to_numpy()

ulaz_train, ulaz_test, oznaka_train, oznaka_test = train_test_split(
    ulazi, oznaka, test_size=0.2, stratify=oznaka, random_state=42
)

skaliraj = StandardScaler()
ulaz_train = skaliraj.fit_transform(ulaz_train)
ulaz_test = skaliraj.transform(ulaz_test)

stablo = DecisionTreeClassifier(random_state=42)
stablo.fit(ulaz_train, oznaka_train)

predikcija = stablo.predict(ulaz_test)

matrica = confusion_matrix(oznaka_test, predikcija)
ConfusionMatrixDisplay(matrica, display_labels=["Slobodna", "Zauzeta"]).plot(cmap="Oranges")
plt.title("Matrica zabune - Stablo odlučivanja")
plt.grid(False)
plt.show()

print("Rezultati klasifikacije:\n")
print(classification_report(oznaka_test, predikcija, target_names=["Slobodna", "Zauzeta"]))

plt.figure(figsize=(12, 6))
plot_tree(stablo, feature_names=["S3_Temp", "S5_CO2"], class_names=["Slobodna", "Zauzeta"], filled=True)
plt.title("Prikaz stabla odlučivanja")
plt.show()
