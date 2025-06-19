fhand = open('song.txt', encoding='utf-8')

counter = {}

for line in fhand:
    line = line.rstrip()

    for rijec in line.split():
        rijec = rijec.lower()

        if rijec in counter:
            counter[rijec] += 1
        else:
            counter[rijec] = 1

fhand.close()

for rijec in counter:
    br = counter[rijec]
    if br == 1:
        print(rijec)