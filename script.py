import datetime
from random import random, seed
from typing import List, Union

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
        return f'Neuron, activation: {self.activation}, error delta: {self.delta}, bias: {self.bias}, weights: {self.weights}'

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
        text = 'Layer: \n'
        for n in self.neurons:
            text += '    ' + repr(n) + '\n'
        return text


class NeuronNetwork:
    def __init__(self, number_of_layers: int, neurons_in_layers: List[int]) -> None:
        # Number of layers include input layer
        assert len(neurons_in_layers) == number_of_layers
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
        # print('Forward propagation...')

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
        # print('Backpropagation...')

        number_of_layers = len(self.layers)

        for index in reversed(range(number_of_layers)):
            layer = self.layers[index]

            if index == number_of_layers - 1:  # last layer
                for i, neuron in enumerate(layer.neurons):
                    error = (neuron.activation - expected_output[i]) * sigmoid_derivative(neuron.activation)
                    neuron.set_error_delta(error)

            else:  # hidden layers
                for i, neuron in enumerate(layer.neurons):
                    next_layer = self.layers[index + 1]
                    error = 0.0

                    for next_neuron in next_layer.neurons:
                        error += (next_neuron.weights[i] * next_neuron.delta)

                    neuron.set_error_delta(error * sigmoid_derivative(neuron.activation))

    def _update_weights(self, row: List[float], learning_rate: float):
        # Call after forward and back propagation
        for i, layer in enumerate(self.layers):
            inputs = row[:-1]  # last row
            if i != 0:
                previous_layer = self.layers[i - 1]
                inputs = [neuron.activation for neuron in previous_layer.neurons]

            for neuron in layer.neurons:
                for input_index in range(len(inputs)):
                    neuron.weights[input_index] -= learning_rate * neuron.delta * inputs[input_index]
                neuron.weights[-1] -= learning_rate * neuron.delta

    def train(self, dataset: List[List[Union[float, int]]], learning_rate: float, repeats: int, number_of_outputs: int):
        for iteration in range(repeats):
            sum_error = 0
            for row in dataset:
                output = self.forward_propagate(row)
                expected = [0 for _ in range(number_of_outputs)]
                expected[row[-1]] = 1
                # print(f'Expected: {expected}')

                sum_error += sum([(expected[i] - output[i]) ** 2 for i in range(len(expected))])
                # print(f'Sum error: {sum_error}')
                self.back_propagate(expected)
                self._update_weights(row, learning_rate)

            print(f'Iteration: {iteration}, learning rate: {learning_rate}, error: {sum_error}')


def main():
    seed(datetime.datetime.now())
    # network = NeuronNetwork(3, [2, 1, 2])  # including input layer which is not exactly a layer
    # print(network)
    dataset = [[2.7810836, 2.550537003, 0],
               [1.465489372, 2.362125076, 0],
               [3.396561688, 4.400293529, 0],
               [1.38807019, 1.850220317, 0],
               [3.06407232, 3.005305973, 0],
               [7.627531214, 2.759262235, 1],
               [5.332441248, 2.088626775, 1],
               [6.922596716, 1.77106367, 1],
               [8.675418651, -0.242068655, 1],
               [7.673756466, 3.508563011, 1]]

    outputs = 2
    network = NeuronNetwork(3, [2, 2, outputs])  # including input layer which is not exactly a layer
    print(network)

    network.train(dataset, 0.5, 20, outputs)
    print(network)

    # network.train()

    # output = network.forward_propagate([1, 0])
    # print(output)
    # print('\n')
    # print(network)
    #
    # expected = [0, 1]
    # network.back_propagate(expected)
    # print('\n')
    # print(network)


main()
