import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

automobili = pd.read_csv('mtcars.csv')

pros_potrosnja = automobili.groupby('cyl')['mpg'].mean()
plt.figure(figsize=(8, 4))
plt.bar(pros_potrosnja.index, pros_potrosnja, color=['navy', 'seagreen', 'crimson'])
plt.xlabel('Broj cilindara')
plt.ylabel('Prosječna potrošnja (mpg)')
plt.title('Prosječna potrošnja po broju cilindara')
plt.show()

plt.figure(figsize=(8, 4))
automobili.boxplot(column='wt', by='cyl', grid=False)
plt.xlabel('Cilindri')
plt.ylabel('Težina (1000 lbs)')
plt.title('Raspon težine po broju cilindara')
plt.show()

plt.figure(figsize=(8, 4))
automobili.boxplot(column='mpg', by='am', grid=False)
plt.xlabel('Tip mjenjača (0 = automatski, 1 = ručni)')
plt.ylabel('Potrošnja (mpg)')
plt.title('Potrošnja po tipu mjenjača')
plt.show()

plt.figure(figsize=(8, 5))
boje = ['red' if m == 0 else 'blue' for m in automobili['am']]
plt.scatter(automobili['hp'], automobili['qsec'], c=boje)
plt.xlabel('Snaga (hp)')
plt.ylabel('Ubrzanje (qsec)')
plt.title('Ubrzanje vs snaga za tip mjenjača')
plt.legend(['Automatski', 'Ručni'], loc='upper right')
plt.show()
