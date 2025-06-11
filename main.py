import random
import math

def function(x):
    return x**2

sample_points = 200
ips = []

for k in range(0, sample_points):
    ips.append(random.choice(range(-100, 100)))

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

# N1 = s(W * N0 + B)

class NeuralNetwork:
    weights = [] # the list of matrices corresponding to the weights
    biases = [] # the list of matrices/vectors corresponding to the biases
    neurons = [] # list of matrices/vectors corresponding to the neuron values

    def __init__(self, layout):
        self.layers = layout # [1, 3, 3, 1]
        for i in range(len(self.layers) - 1): 
            n1 = self.layers[i]
            n2 = self.layers[i+1]
            self.weights.append(Matrix(
                [[random.uniform(-0.5, 0.5) 
                  for _ in range(n1)] for _ in range(n2)]
                ))
            
            self.biases.append(Matrix(
                [[random.uniform(-5, 5)] for _ in range(n2)]
                ))
            self.neurons.append(Matrix(
                [[0] for _ in range(n2)]
            ))

    @staticmethod
    def add_noise(m, percent):
        new = Matrix(m.ls)
        for i in range(m.rows):
            for j in range(m.cols):
                new.ls[i][j] += percent*m.ls[i][j]

        return new
                 
    
    @staticmethod
    def sigmoid(m): # first activation function
        m1 = m.ls
        for i in range(m.rows):
            for j in range(m.cols):
                #m1[i][j] = 1 / (1 + math.exp(-m1[i][j]))
                m1[i][j] = max(0, m1[i][j])
        return Matrix(m1)
    
    def compute(self, inp):
        i = 0
        l = [inp] + self.neurons[:-1] # l -> [inp, layer1, layer2, ...] but not the output layer
        for weight, bias in zip(self.weights, self.biases):
            self.neurons[i] = self.sigmoid(weight * l[i] + bias)
            i += 1

amount = 10
nns = [NeuralNetwork([1, 2, 1]) for _ in range(amount)]

mc = [math.inf, 0]


for _ in range(100):
    for n in range(amount):
        cost = 0
        for i, inp in enumerate(ips):
            nn = nns[n]
            inpu = Matrix([[inp]])
            nn.compute(inpu)
            output = nn.neurons[-1]
            cost += abs(y[i] - output.ls[0][0])
            if cost < mc[0]:
                mc[0] = cost
                mc[1] = n

    nns = [nns[n] for _ in range(100)]
    for k in range(1, amount):
        p = random.choice([-1, 1]) * k/4
        for w in range(len(nns[k].weights)):
            nns[k].weights[w] = nns[k].add_noise(nns[k].weights[w], p)
        for b in range(len(nns[k].biases)):
            nns[k].biases[b] = nns[k].add_noise(nns[k].biases[b], p)

point = int(input("enter a point : "))
nns[0].compute(Matrix([[point]]))
print(nns[0].neurons[-1].ls[0][0])
