import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def ciljna_funkcija(x):
    return 1.6345 - 0.6235 * np.cos(0.6067 * x) - 1.3501 * np.sin(0.6067 * x) \
           - 1.1622 * np.cos(2 * x * 0.6067) - 0.9443 * np.sin(2 * x * 0.6067)

def dodaj_sumu(y):
    np.random.seed(14)
    raspon = np.max(y) - np.min(y)
    return y + 0.1 * raspon * np.random.normal(0, 1, len(y))

x = np.linspace(1, 10, 50)
y_true = ciljna_funkcija(x)
y_mjereno = dodaj_sumu(y_true)

x = x[:, np.newaxis]
y_mjereno = y_mjereno[:, np.newaxis]

# Polinomne značajke
stupanj = 15
poly = PolynomialFeatures(degree=stupanj)
x_pol = poly.fit_transform(x)

# Treniranje i testiranje
np.random.seed(12)
perm = np.random.permutation(len(x_pol))
train_idx = perm[:int(0.7 * len(x_pol))]
test_idx = perm[int(0.7 * len(x_pol)) + 1:]

xtrain = x_pol[train_idx]
ytrain = y_mjereno[train_idx]
xtest = x_pol[test_idx]
ytest = y_mjereno[test_idx]

model = lm.LinearRegression()
model.fit(xtrain, ytrain)

ytest_pred = model.predict(xtest)
mse = mean_squared_error(ytest, ytest_pred)
print(f"Srednja kvadratna pogreška (MSE) na test skupu: {mse:.4f}")

# Graf 1: Predikcija vs test
plt.figure(1)
plt.plot(xtest[:, 1], ytest_pred, 'og', label='Predikcija')
plt.plot(xtest[:, 1], ytest, 'or', label='Stvarne vrijednosti')
plt.title("Test skup vs predikcija")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc=4)
plt.grid(True)
plt.tight_layout()
plt.show()

# Graf 2: Model vs stvarna funkcija
plt.figure(2)
plt.plot(x[:, 0], y_true, label='Stvarna funkcija f(x)')
plt.plot(x[:, 0], model.predict(x_pol), 'r-', label=f'Polinomni model (stupanj={stupanj})')
plt.plot(xtrain[:, 1], ytrain, 'ok', label='Uzorci za učenje')
plt.title("Model vs stvarna funkcija")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc=4)
plt.grid(True)
plt.tight_layout()
plt.show()
