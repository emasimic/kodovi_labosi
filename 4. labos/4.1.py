import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error

def ciljna_funkcija(x):
    y = 1.6345 - 0.6235 * np.cos(0.6067 * x) - 1.3501 * np.sin(0.6067 * x) \
        - 1.1622 * np.cos(2 * x * 0.6067) - 0.9443 * np.sin(2 * x * 0.6067)
    return y

def dodaj_sumu(y):
    np.random.seed(14)
    raspon = np.max(y) - np.min(y)
    y_sum = y + 0.1 * raspon * np.random.normal(0, 1, len(y))
    return y_sum

x = np.linspace(1, 10, 100)
y_stvarno = ciljna_funkcija(x)
y_mjereno = dodaj_sumu(y_stvarno)

plt.figure(1)
plt.plot(x, y_mjereno, 'ok', label='Mjereno')
plt.plot(x, y_stvarno, label='Stvarno')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)
plt.show()

np.random.seed(12)
perm = np.random.permutation(len(x))
ind_train = perm[:int(0.7 * len(x))]
ind_test = perm[int(0.7 * len(x)) + 1:]

x = x[:, np.newaxis]
y_mjereno = y_mjereno[:, np.newaxis]

x_tren = x[ind_train]
y_tren = y_mjereno[ind_train]
x_test = x[ind_test]
y_test = y_mjereno[ind_test]

plt.figure(2)
plt.plot(x_tren, y_tren, 'ob', label='Skup za učenje')
plt.plot(x_test, y_test, 'or', label='Test skup')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)
plt.show()

model = lm.LinearRegression()
model.fit(x_tren, y_tren)

print('Model: y_hat = Theta0 + Theta1 * x')
print('y_hat =', model.intercept_, '+', model.coef_, '* x')

y_test_p = model.predict(x_test)
mse = mean_squared_error(y_test, y_test_p)
print("MSE na test skupu:", mse)

plt.figure(3)
plt.plot(x_test, y_test_p, 'og', label='Predikcija')
plt.plot(x_test, y_test, 'or', label='Stvarne vrijednosti')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)
plt.show()
