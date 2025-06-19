import urllib.request
import pandas as pd
import matplotlib.pyplot as plt

url = 'http://iszz.azo.hr/iskzl/rs/podatak/export/xml?postaja=160&polutant=5&tipPodatka=0&vrijemeOd=01.01.2017&vrijemeDo=31.12.2017'

data = urllib.request.urlopen(url).read()
df = pd.read_xml(data)

df = df[['vrijednost', 'vrijeme']]
df.columns = ['koncentracija', 'vrijeme']
df['koncentracija'] = df['koncentracija'].astype(float)
df['vrijeme'] = pd.to_datetime(df['vrijeme'], utc=True)

top3 = df.nlargest(3, 'koncentracija')

print("Tri datuma s najvećom koncentracijom PM10 u 2017:")
print(top3[['vrijeme', 'koncentracija']])

plt.figure(figsize=(10, 5))
plt.plot(df['vrijeme'], df['koncentracija'], label='PM10 (µg/m³)')
plt.title('Koncentracija PM10 u Osijeku - 2017')
plt.xlabel('Datum')
plt.ylabel('PM10 (µg/m³)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()
