fhand = open('SMSSpamCollection.txt', encoding='utf-8')
usklicnaRecenica = 0
brHamRijeci = 0
brHamPoruka = 0
brSpamPoruka = 0
brSpamRijeci = 0

for line in fhand:
    line = line.rstrip()

    if not line:
        continue

    if line.startswith('ham'):
        oznaka = 'ham'
        poruka = line[4:].strip()
    elif line.startswith('spam'):
        oznaka = 'spam'
        poruka = line[5:].strip()
    else:
        continue

    brRijeci = len(poruka.split())


    if oznaka == 'ham':
        brHamRijeci += brRijeci
        brHamPoruka += 1
    elif oznaka == 'spam':
        brSpamRijeci += brRijeci
        brSpamPoruka += 1

        if poruka.endswith('!'):
            usklicnaRecenica += 1

fhand.close()

if brHamPoruka > 0:
    prosjekHam = brHamRijeci / brHamPoruka
else:
    prosjekHam = 0
if brSpamPoruka > 0:
    prosjekSpam = brSpamRijeci / brSpamPoruka
else:
    prosjekSpam = 0

print(prosjekSpam)
print(usklicnaRecenica)
print(prosjekHam)
