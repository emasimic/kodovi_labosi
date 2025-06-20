import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, max_error

podaci = pd.read_csv('cars_processed.csv')
print(podaci.info())

podaci = pd.get_dummies(podaci, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)

X = podaci.drop(columns=['name', 'selling_price', 'seats'])
y = podaci['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=300)

skaliranje = StandardScaler()
X_train_scaled = skaliranje.fit_transform(X_train)
X_test_scaled = skaliranje.transform(X_test)

reg_model = LinearRegression()
reg_model.fit(X_train_scaled, y_train)

train_predictions = reg_model.predict(X_train_scaled)
test_predictions = reg_model.predict(X_test_scaled)

print(r2_score(y_test, test_predictions))
print(np.sqrt(mean_squared_error(y_test, test_predictions)))
print(max_error(y_test, test_predictions))
print(mean_absolute_error(y_test, test_predictions))

plt.figure(figsize=(13, 10))
sns.regplot(x=test_predictions, y=y_test, line_kws={'color': 'green'})
plt.xlabel('Predviđena vrijednost')
plt.ylabel('Stvarna vrijednost')
plt.title('Usporedba predikcije i stvarne cijene')
plt.grid(True)
plt.show()
