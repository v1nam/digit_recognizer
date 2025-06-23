import random
import math
import time
from copy import deepcopy
from matplotlib import pyplot as plt

amount = 40 # number of neural networks (we have "amount" number of neural networks) 
iterat_amount = 400
n1 = iterat_amount/4
n2 = iterat_amount/2
n3 = iterat_amount/1.5
def function(x):
    return math.sin(x)

sample_points = 200
s, e = -4, 4
ips = []

for k in range(0, sample_points):
    ips.append(s + (e-s)*(k/sample_points))

y = [function(i) for i in ips]


class Matrix:
    def __init__(self, m):
        self.rows = len(m)
        self.cols = len(m[0])
        self.ls = m
    
    def __mul__(self, m2):
        if self.cols != m2.rows:
            return "Matrices cannot be multiplied"
        
        prod = [[0 for _ in range(m2.cols)] for _ in range(self.rows)]
        
        for i in range(self.rows):
            for j in range(m2.cols):
                prod[i][j] = sum(self.ls[i][c]*m2.ls[c][j] for c in range(self.cols))
        
        return Matrix(prod)

    def __add__(self, m2):
        if (self.rows, self.cols) != (m2.rows, m2.cols):
            return "Matrices cannot be added"
    
        ad = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(self.cols):
                ad[i][j] = self.ls[i][j] + m2.ls[i][j]
        
        return Matrix(ad)
    
    def __sub__(self, m2):
        for i in range(m2.rows):
            for j in range(m2.cols):
                m2.ls[i][j] *= -1
        
        return self + m2
    
    def __repr__(self):
        return str(self.ls)

# N1 = s(W * N0 + B)

def add_noise_w(m, k):
    new = Matrix(m.ls)
    for i in range(m.rows):
        for j in range(m.cols):
            if random.choice(range(10)) == 1: # 1/10 chance to mutate the weight
                if k <= n1: 
                    new.ls[i][j] += random.choice((-1, 1))*random.gauss(0, 1.7)
                elif k <= n2: 
                    new.ls[i][j] += random.choice((-1, 1))*random.gauss(0, 0.85)
                elif k <= n3: 
                    new.ls[i][j] += random.choice((-1, 1))*random.gauss(0, 0.5)
                else:
                    new.ls[i][j] += random.choice((-1, 1))*random.gauss(0, 0.1)

    return new

def add_noise_b(m, k): #taking number of iterations for dynamic mutations
    new = Matrix(m.ls)
    
    for i in range(m.rows):
        for j in range(m.cols):
            if random.choice(range(10)) == 1:
                if k <= n1:
                    new.ls[i][j] += random.choice((-1, 1))*random.gauss(0, 1.4)
                elif k <= n2:
                    new.ls[i][j] += random.choice((-1, 1))*random.gauss(0, 1)
                elif k <= n3:
                    new.ls[i][j] += random.choice((-1, 1))*random.gauss(0, 0.8)
                else:
                    new.ls[i][j] += random.choice((-1, 1))*random.gauss(0, 0.5)

    return new

class NeuralNetwork:
    def __init__(self, layout):
        self.layers = layout # [1, 3, 3, 1]
        self.weights = [] # the list of matrices corresponding to the weights
        self.biases = [] # the list of matrices/vectors corresponding to the biases
        self.neurons = [] # list of matrices/vectors corresponding to the neuron values
        for i in range(len(self.layers) - 1): 
            n1 = self.layers[i]
            n2 = self.layers[i+1]
            self.weights.append(Matrix(
                [[random.uniform(-4.472, 4.472) 
                  for _ in range(n1)] for _ in range(n2)]
                ))
            
            self.biases.append(Matrix(
                [[random.uniform(-1, 1)] for _ in range(n2)]
                ))
            self.neurons.append(Matrix(
                [[0] for _ in range(n2)]
            ))
    
    @staticmethod
    def LeakyRelU(m): # first activation function
        m1 = m.ls
        for i in range(m.rows):
            for j in range(m.cols):
                # m1[i][j] = 1 / (1 + math.exp(-m1[i][j]))
                m1[i][j] = max(0.01*m1[i][j], m1[i][j])

        return Matrix(m1)
    
    def compute(self, inp):
        i = 0
        # l = [inp] + self.neurons[:-1] # l -> [inp, layer1, layer2, ...] but not the output layer
        for weight, bias in zip(self.weights, self.biases):
            w = weight * (inp if i == 0 else self.neurons[i-1]) + bias
            self.neurons[i] = self.LeakyRelU(w) if i != (len(self.biases)-1) else w
            i += 1
        return self.neurons[-1]



nns = []
for _ in range(amount):
    v = NeuralNetwork([1, 10, 20, 10, 1])
    nns.append(v)

mc = [math.inf, 0]
for j in range(iterat_amount): #number of evolutions
    for n in range(amount):
        cost = 0
        nn = nns[n]
        for i, inp in enumerate(ips): # (index, value)
            inpu = Matrix([[inp]])
            output = nn.compute(inpu)
            cost += abs(y[i] - output.ls[0][0])
        print(cost, end=", ")
        if cost < mc[0]:
            mc[0] = cost
            nns[0] = deepcopy(nn)
            mc[1] = n
    print("\n")
    print(mc, j)
    print("-"*100)
    
    for k in range(1, amount):
        nns[k] = deepcopy(nns[0])
    
    for k in range(1, amount): #  going through all the neural networks to mutate them except the zeroeth one
        for w in range(len(nns[k].weights)):
            nns[k].weights[w] = add_noise_w(nns[k].weights[w], j)
        for b in range(len(nns[k].biases)):
            nns[k].biases[b] = add_noise_b(nns[k].biases[b], j)
    
    
ips = [s + (i/(sample_points*10))*(e - s) for i in range(sample_points*10)]
plt.plot(ips, list(map(function, ips)))
plt.plot(ips, list(map(lambda i: nns[0].compute(Matrix([[i]])).ls[0][0], ips)))
plt.show()