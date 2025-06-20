import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Definicija ciljne (skrivene) funkcije
def ciljna_funkcija(x):
    return (1.6345 
            - 0.6235 * np.cos(0.6067 * x) 
            - 1.3501 * np.sin(0.6067 * x) 
            - 1.1622 * np.cos(2 * 0.6067 * x) 
            - 0.9443 * np.sin(2 * 0.6067 * x))

# Funkcija za dodavanje šuma mjerenjima
def dodaj_smetnje(y):
    np.random.seed(14)
    raspon = np.max(y) - np.min(y)
    smetnje = 0.1 * raspon * np.random.normal(0, 1, size=len(y))
    return y + smetnje

# Funkcija za polinomnu regresiju s danim stupnjem
def treniraj_model(stupanj, x, y, udio_trening=0.7):
    poly_gen = PolynomialFeatures(degree=stupanj)
    x_prosireno = poly_gen.fit_transform(x)

    y = y.flatten()
    np.random.seed(12)
    permutacija = np.random.permutation(len(x_prosireno))
    granica = int(udio_trening * len(x_prosireno))

    x_tr = x_prosireno[permutacija[:granica]]
    y_tr = y[permutacija[:granica]]
    x_ts = x_prosireno[permutacija[granica:]]
    y_ts = y[permutacija[granica:]]

    model = LinearRegression()
    model.fit(x_tr, y_tr)

    mse_tr = mean_squared_error(y_tr, model.predict(x_tr))
    mse_ts = mean_squared_error(y_ts, model.predict(x_ts))

    return model, poly_gen, mse_tr, mse_ts

# Priprema podataka
broj_uzoraka = 50
x_os = np.linspace(1, 10, broj_uzoraka).reshape(-1, 1)
y_idealno = ciljna_funkcija(x_os)
y_mjereno = dodaj_smetnje(y_idealno).reshape(-1, 1)

# Testiramo više modela s različitim stupnjevima
stupnjevi = [2, 6, 15]
mse_trening = []
mse_test = []

plt.figure(figsize=(9, 6))
plt.plot(x_os, y_idealno, 'k--', label='Stvarna funkcija')

for stupanj in stupnjevi:
    model, poly, mse_tr, mse_ts = treniraj_model(stupanj, x_os, y_mjereno)
    mse_trening.append(mse_tr)
    mse_test.append(mse_ts)

    x_trans = poly.transform(x_os)
    y_pred = model.predict(x_trans)
    plt.plot(x_os, y_pred, label=f'Polinom (stupanj {stupanj})')

plt.scatter(x_os, y_mjereno, color='gray', alpha=0.6, label='Šumom pogođeni podaci')
plt.xlabel('x vrijednosti')
plt.ylabel('y vrijednosti')
plt.title('Usporedba polinomnih modela')
plt.legend()
plt.tight_layout()
plt.show()

print("Pogreške na skupu za učenje:", mse_trening)
print("Pogreške na test skupu:", mse_test)
