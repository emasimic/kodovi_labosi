brojevi = []
while True:
    try:
        unos = input("unesi broj ili napisi done ako zelis zavrsiti: ")
        if unos.lower() == 'done':
            break
        provjera = float(unos)
        brojevi.append(provjera)
    except:
        print("nisi unio/unijela broj")

broj = len(brojevi)
srednjaVrijednost = sum(brojevi)/broj
minVrijednost = min(brojevi)
maxVrijednost = max(brojevi)

brojevi.sort()

print(broj)
print(srednjaVrijednost)
print(minVrijednost)
print(maxVrijednost)
print(brojevi)