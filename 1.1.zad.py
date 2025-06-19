def total_euro(sati, satnica):
    return sati * satnica

sati = float(input("Radni sati: "))
satnica = float(input("Eura/h: "))
ukupno = total_euro(sati, satnica)

print(f"Ukupno: {ukupno} eura")