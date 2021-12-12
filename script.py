import datetime
from random import random, seed
from typing import List

from numpy.ma import exp


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


class Neuron:
    def __init__(self, bias: float, weights: List[float]) -> None:
        self.bias = bias
        self.weights = weights
        self.activation = 0

    def __repr__(self):
        return f'Neuron, activation: {self.activation}, bias: {self.bias}, weights: {self.weights}'

    def set_activation(self, inputs: List[float]) -> None:
        # activation = SUM(weight i-1 * input i-1) + bias

        activation = self.bias
        for i in range(len(self.weights)):
            activation += self.weights[i] * inputs[i]

        self.activation = activation


class NeuronLayer:
    def __init__(self, number_of_neurons: int, previous_layer_length: int) -> None:
        self.neurons: List[Neuron] = []
        for i in range(number_of_neurons):
            start_bias = random()
            weights = []
            for j in range(previous_layer_length):
                weights.append(random())
            self.neurons.append(Neuron(start_bias, weights))

    def __repr__(self):
        text = 'Layer: '
        for n in self.neurons:
            text += repr(n) + ', '
        return text


class NeuronNetwork:
    def __init__(self, number_of_layers: int, neurons_in_layers: List[int]) -> None:
        assert len(neurons_in_layers) == number_of_layers
        self.layers: List[NeuronLayer] = []
        for i in range(number_of_layers):
            if i == 0:
                continue
            previous_layer_length = neurons_in_layers[i - 1]
            self.layers.append(NeuronLayer(neurons_in_layers[i], previous_layer_length))

    def __repr__(self):
        text = 'Network:\n'
        for layer in self.layers:
            text += repr(layer) + '\n'

        return text


def main():
    seed(datetime.datetime.now())
    network = NeuronNetwork(3, [2, 1, 2])
    print(network)


main()
