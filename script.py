from random import random, seed
from typing import List


class Neuron:
    def __init__(self, bias: int, weights: List[int]) -> None:
        self.bias = bias
        self.weights = weights

    def __repr__(self):
        return f'Neuron, bias: {self.bias}, weights: {self.weights}'


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


def main():
    seed(1)
    network = NeuronNetwork(3, [2, 1, 2])
    for layer in network.layers:
        print(layer)

main()
