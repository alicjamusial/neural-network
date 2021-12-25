from datetime import datetime
import pickle
import time
from random import random, seed
from typing import List, Union

from numpy.ma import exp
from matplotlib import pyplot as plt
import numpy as np


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
        activation = self.bias
        for i in range(len(self.weights)):
            activation += self.weights[i] * inputs[i]

        activation = activation
        self.activation = sigmoid(activation)

    def set_error_delta(self, error_delta: float) -> None:
        self.delta = error_delta


class NeuronLayer:
    def __init__(self, number_of_neurons: int, previous_layer_length: int) -> None:
        self.neurons: List[Neuron] = []

        for i in range(number_of_neurons):
            start_bias = (random() - 0.5) * 2
            weights = []
            for j in range(previous_layer_length):
                weights.append((random() - 0.5) * 2)
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
                    # neuron.bias -= learning_rate * 1 * neuron.delta
                neuron.weights[-1] -= learning_rate * neuron.delta

    def _randomize_biases(self):
        for i, layer in enumerate(self.layers):
            for neuron in layer.neurons:
                neuron.bias = random()

    def train(self, dataset: List[List[Union[float, int]]], learning_rate: float, repeats: int, number_of_outputs: int) -> List[int]:
        all_errors = []
        prev_error = 10000
        for iteration in range(repeats):
            sum_error = 0
            for row in dataset:
                output = self.forward_propagate(row)
                expected = [0 for _ in range(number_of_outputs)]
                expected[row[-1]] = 1

                sum_error += sum([(expected[i] - output[i]) ** 2 for i in range(len(expected))])
                self.back_propagate(expected)
                self._update_weights(row, learning_rate)

            if len(all_errors) > 0:
                prev_error = all_errors[-1:][0]

            all_errors.append(sum_error)

            # print(f'Iteration: {iteration}, learning rate: {learning_rate}, error: {sum_error}', flush=True)
            # if len(all_errors) > 0 and abs(prev_error - sum_error) < 5:
            #     print(f'Diff too small, leaving this shit', flush=True)
            #     self._randomize_biases()
            #     # break

            # if sum_error < 20:
            #     learning_rate = 0.4
            # if sum_error < 10:
            #     learning_rate = 0.1
            # if sum_error < 3:
            #     learning_rate = 0.05
            if iteration == 99:
                print(f'Iteration: {iteration}, learning rate: {learning_rate}, error: {sum_error}', flush=True)
        return all_errors

    def predict(self, row: List[float]):
        output = self.forward_propagate(row)
        # print(f'Probabilities: {output}')
        max_probability = max(output)
        return output.index(max_probability)

#
# def main():
#     seed(datetime.now())
#
#     xor = [[0, 0, 0],
#            [0, 1, 1],
#            [1, 0, 1],
#            [1, 1, 0]]
#
#     outputs = 2
#     network = NeuronNetwork(3, [2, 4, outputs])  # including input layer which is not exactly a layer
#     print(network)
#
#     errors = network.train(xor, 0.1, 8000, outputs)
#     print(network)
#
#     dataset_to_predict = [[0, 0, 0],
#                           [0, 1, 1],
#                           [1, 0, 1],
#                           [1, 1, 0]]
#
#     for row in dataset_to_predict:
#         prediction = network.predict(row)
#         print(f'Expected={row[-1]}, Got={prediction}')
#

# main()


def plot_image(arr):
    two_d = (np.reshape(arr, (28, 28))).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest', cmap='Greys_r')
    return plt


def plot_errors(errors):
    x = [i for i in range(len(errors))]

    plt.plot(x, errors)
    plt.show()


def read_mnist_ready():
    with open('mnist_results/mnist_ready', 'rb') as fp:
        return pickle.load(fp)


def read_mnist_raw():
    with open('mnist_results/mnist_raw', 'rb') as fp:
        return pickle.load(fp)


def read_saved_neuron_network(file: str):
    with open(f'networks/{file}', 'rb') as fp:
        return pickle.load(fp)


# for i, img in enumaerate(new_images[120:130]):
#     gen_image(img).show()
#     print(all_labels[120 + i])
# new_images = read_mnist_raw()
# gen_image(new_images[126]).show()
# print(all_labels[126])
# gen_image(new_images[4098]).show()
# print(all_labels[4098])

#
# seed(datetime.now())
# images_dataset = read_mnist_ready()
#
# outputs = 10
# network = NeuronNetwork(3, [784, 16, outputs])
#
# every_error_ever = []
#
# for i in range(20):
#     dataset = images_dataset[i*100:(i*100)+100]
#     all_errors_sums = network.train(dataset, 0.8, 100, outputs)
#     every_error_ever.extend(all_errors_sums)
#
#     dataset_to_predict = images_dataset[2500:2550]
#     errors = 0
#     for row in dataset_to_predict:
#         prediction = network.predict(row)
#         if row[-1] != prediction:
#             errors += 1
#
#     print(f'Errors ratio: {errors}/{len(dataset_to_predict)}')
#
# plot_errors(every_error_ever)
#
# # network = read_saved_neuron_network('14-12-2021-17-31-54')
#
# dataset_to_predict = images_dataset[2500:2550]
# errors = 0
# for row in dataset_to_predict:
#     prediction = network.predict(row)
#     if row[-1] != prediction:
#         errors += 1
#
# print(f'Errors ratio: {errors}/{len(dataset_to_predict)}')
#
# # should = input('Should save network?')
# # if should == 'y':
# with open(f'networks/{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}', 'wb') as fp:
#     pickle.dump(network, fp)
# with open(f'networks/{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}-errors-sums', 'wb') as fp:
#     pickle.dump(all_errors_sums, fp)
