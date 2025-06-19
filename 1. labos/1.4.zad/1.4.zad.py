dat = input("unesi ime datoteke: ")

try:
    fhand = open(dat)
    ukupnaPouzdanost = 0
    brLinija = 0

    for line in fhand:
        line = line.rstrip()
        if line.startswith("X-DSPAM-Confidence:"):
            for rijec in line.split():
                try:
                    br = float(rijec)
                    ukupnaPouzdanost += br
                    brLinija += 1
                    break
                except:
                    continue
    fhand.close()

    srednjaVrijednost = ukupnaPouzdanost / brLinija

    if brLinija > 0:
        print("Srednja vrijednost pouzdanosti je: ", srednjaVrijednost)
    else:
        print("Nema pronadjenih vrijednosti")
except:
    print("Datoteka nije pronadjena")