import pickle
from matplotlib import pyplot as plt


def read_errors_neuron_network(file: str):
    with open(f'networks/{file}-errors-sums', 'rb') as fp:
        return pickle.load(fp)


colors = ['red', 'green', 'blue', 'pink', 'cyan', 'black']

def plot_errors(errors, color):
    x = [i for i in range(len(errors[-20:]))]

    plt.plot(x, errors[-20:], color=colors[color])

files = ['14-12-2021-19-44-11', '14-12-2021-19-57-43', '14-12-2021-20-11-19', '14-12-2021-20-24-44', '14-12-2021-20-36-09', '14-12-2021-20-48-40']

for i, file in enumerate(files):
    errors = read_errors_neuron_network(file)
    plot_errors(errors, i)


plt.show()