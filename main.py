# from matplotlib import pyplot as plt
import random

def function(x):
    return x**2


ips = list(range(-100, 100))
y = [function(i) for i in ips]


# func(1, 3) -> [ r] (1x3)


class Neuron:
    value = 0
    def __init__(self, inp=False):
        self.bias =  random.uniform(-10, 10)
        if inp:
            self.value = ...

    def update(self, new):
        self.value = new

class NeuralNetwork:
    inp = 1
    layers = [3, 1]
    weights = []
    neurons = [Neuron(inp=True) for i in inp] + [[Neuron() for n in range(layer)] for layer in layers]

    def __init__(self):
        for l in range(len(self.layers) - 1):
            l1 = self.layers[l]
            l2 = self.layers[l + 1]
            
            self.weights.append(
                [[random.uniform(0, 1) for _ in range(l2)] for _ in range(l1)]
            )

    def activate(self):
        for nl in self.neurons:
            for neuron in nl:
                new_value = 
                neuron.update(new_value)

nn = NeuralNetwork()

#training
for i, inp in enumerate(ips):
    y1 = y[i]
    
