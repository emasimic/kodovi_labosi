import pandas as pd

podaci = pd.read_csv('mtcars.csv')

najbolji_ekonomici = podaci.nsmallest(5, 'mpg')
print(najbolji_ekonomici)

osmicyl = podaci[podaci['cyl'] == 8]
najlosiji_8cyl = osmicyl.nlargest(3, 'mpg')
print(najlosiji_8cyl)

sestcilindrasa = podaci.loc[podaci['cyl'] == 6, 'mpg'].mean()

cetvorka_wt = podaci[(podaci['cyl'] == 4) & (podaci['wt'].between(2.0, 2.2))]
pros_potrosnja_cetvorka = cetvorka_wt['mpg'].mean()
print(pros_potrosnja_cetvorka)

mjenjaci = podaci['am'].value_counts()
print("0 = automatski, 1 = ručni")
print(mjenjaci)

broj_jakih_auto = podaci[(podaci['am'] == 0) & (podaci['hp'] > 100)].shape[0]
print(broj_jakih_auto)

podaci['masa_kg'] = podaci['wt'] * 453.592
print(podaci[['masa_kg']])
