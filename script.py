import datetime
from random import random, seed
from typing import List

from numpy.ma import exp


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def sigmoid_derivative(value: float) -> float:
    return value * (1.0 - value)


class Neuron:
    def __init__(self, bias: float, weights: List[float]) -> None:
        self.bias = bias
        self.weights = weights
        self.activation = 0
        self.delta = 0

    def __repr__(self):
        return f'Neuron, activation: {self.activation}, bias: {self.bias}, weights: {self.weights}'

    def compute_and_set_activation(self, inputs: List[float]) -> None:
        # activation = SUM(weight i-1 * input i-1) + bias

        activation = self.bias
        for i in range(len(self.weights)):
            activation += self.weights[i] * inputs[i]

        self.activation = sigmoid(activation)

    def set_error_delta(self, error_delta: float) -> None:
        self.delta = error_delta


class NeuronLayer:
    def __init__(self, number_of_neurons: int, previous_layer_length: int) -> None:
        self.neurons: List[Neuron] = []
        # if number_of_neurons == 1:
        #     start_bias = 0.763774618976614
        #     weights = [0.13436424411240122, 0.8474337369372327]
        #     self.neurons.append(Neuron(start_bias, weights))
        #
        # if number_of_neurons == 2:
        #     start_bias = 0.49543508709194095
        #     weights = [0.2550690257394217]
        #     self.neurons.append(Neuron(start_bias, weights))
        #
        #     start_bias = 0.651592972722763
        #     weights = [0.4494910647887381]
        #     self.neurons.append(Neuron(start_bias, weights))

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
        # Number of layers include input layer
        assert len(neurons_in_layers) == number_of_layers
        self.input: List[float] = []
        self.input_len = neurons_in_layers[0]
        self.layers: List[NeuronLayer] = []

        for i in range(number_of_layers):
            if i == 0:
                continue
            previous_layer_length = neurons_in_layers[i - 1]
            self.layers.append(NeuronLayer(neurons_in_layers[i], previous_layer_length))

    def __repr__(self):
        text = 'Network:\n'
        text += f'Input layer: {self.input_len} values\n'
        for layer in self.layers:
            text += repr(layer) + '\n'

        return text

    def forward_propagate(self, data_input: List[float]) -> List[float]:
        print('Forward propagation...')
        assert len(data_input) == self.input_len

        self.input = data_input.copy()

        for layer in self.layers:
            new_input = []
            for neuron in layer.neurons:
                neuron.compute_and_set_activation(data_input)
                new_input.append(neuron.activation)
            data_input = new_input

        output = data_input  # output is last input, magic
        return output

    def back_propagate(self, expected_output: List[float]):
        # Calculate an error for each output neuron (last layer)
        # which will give us input to propagate backwards to previous layers
        print('Backpropagation...')

        number_of_layers = len(self.layers)

        for index in reversed(range(number_of_layers)):
            layer = self.layers[index]

            if index == number_of_layers - 1:  # last layer
                for i, neuron in enumerate(layer.neurons):
                    error = (neuron.activation - expected_output[i]) * sigmoid_derivative(neuron.activation)
                    neuron.set_error_delta(error)

            else:  # hidden layers
                for i, neuron in enumerate(layer.neurons):
                    next_layer = self.layers[index+1]
                    error = 0.0

                    for next_neuron in next_layer.neurons:
                        error += (next_neuron.weights[i] * next_neuron.delta)

                    neuron.set_error_delta(error * sigmoid_derivative(neuron.activation))

def main():
    seed(datetime.datetime.now())
    network = NeuronNetwork(3, [2, 1, 2])  # including input layer which is not exactly a layer
    print(network)

    output = network.forward_propagate([1, 0])
    print(output)
    print('\n')
    print(network)


main()