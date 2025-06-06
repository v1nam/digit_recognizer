import random

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
            self.neurons.append(None)

    
    
    @staticmethod
    def RelU(m): # first activation function
        m1 = m.ls
        for i in range(m.rows):
            for j in range(m.cols):
                m1[i][j] = max(0, m1[i][j])
        return Matrix(m1)
    
    def compute(self, inp):
        i = 0
        l = [inp] + self.neurons[:-1] # l -> [inp, layer1, layer2, ...] but not the output layer
        for weight, bias in zip(self.weights, self.biases):
            self.neurons[i] = self.RelU(weight * l[i] + bias)  # missing activation function
            i += 1

nn = NeuralNetwork([1, 3, 3, 1])