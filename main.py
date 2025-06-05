import random

def function(x):
    return x**2


sample_points = 200
ips = []

for k in range(0, sample_points):
    ips.append(random.choice(range(-100, 100)))

y = [function(i) for i in ips]


#(AB)ij = sum(a)

def mat_mult(m1, m2):
    m1r = len(m1)
    m1c = len(m1[0])

    m2r = len(m2)
    m2c = len(m2[0])

    if m1c != m2r:
        return "Matrices cannot be multiplied"

    prod = [[0 for _ in range(m2c)] for _ in range(m1r)]
    
    for i in range(m1r):
        for j in range(m2c):
            prod[i][j] = sum(m1[i][c]*m2[c][j] for c in range(m1c))
            
    return prod

p = mat_mult([[6, 5, 4], [2, 3, 1]], [[3, 3], [5, 6], [7, 8]])
print(p)