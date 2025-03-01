def check(x: str, file: str):
    f = open(file, mode='w')
    d = dict()
    words = x.lower().split()
    for s in set(words):
        d[s] = words.count(s)
    for key, val in sorted(list(d.items()), key=lambda l: l[0]):
        f.write(key + ' ' + str(val) + '\n')