import pickle
from matplotlib import pyplot as plt


def read_errors_neuron_network(file: str):
    with open(f'networks/{file}-errors-sums', 'rb') as fp:
        return pickle.load(fp)


colors = ['red', 'green', 'blue', 'pink', 'cyan', 'black']

def plot_errors(errors, color):
    x = [i for i in range(len(errors))]

    plt.plot(x, errors, color=colors[color])

files = ['14-12-2021-18-13-45', '14-12-2021-18-25-54', '14-12-2021-18-38-48']

for i, file in enumerate(files):
    errors = read_errors_neuron_network(file)
    plot_errors(errors, i)


plt.show()