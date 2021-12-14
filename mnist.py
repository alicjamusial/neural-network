import pickle
import time

import numpy as np


def open_and_save_mnist():
    now = time.time()

    with open('mnist/train-labels.idx1-ubyte', 'rb') as f:
        labels_data = f.read()

    magic_number_labels = int.from_bytes(labels_data[0:4], 'big')
    number_of_labels = int.from_bytes(labels_data[4:8], 'big')

    all_labels = labels_data[8:]

    with open('mnist/train-images.idx3-ubyte', 'rb') as f:
        data = f.read()

    magic_number = int.from_bytes(data[0:4], 'big')
    number_of_images = int.from_bytes(data[4:8], 'big')
    number_of_rows = int.from_bytes(data[8:12], 'big')
    number_of_cols = int.from_bytes(data[12:16], 'big')

    images_raw = data[16:]
    images = []

    size = number_of_cols * number_of_rows  # 784

    for i in range(number_of_images):
        images.append(images_raw[(i*size):(i*size + size)])

    new_images = []  # numpy array

    for image in images:
        image = [x for x in image]
        a = np.array(image)
        nslices = number_of_rows
        a.reshape((nslices, -1))
        new_images.append(a)

    with open('mnist_results/mnist_raw', 'wb') as raw:
        pickle.dump(new_images, raw)

    images_dataset = []
    for i, image in enumerate(images):
        image = [x/255 for x in image]
        image.append(all_labels[i])
        images_dataset.append(image)

    with open('mnist_results/mnist_ready', 'wb') as ready:
        pickle.dump(images_dataset, ready)

    print(f'Operation took: {time.time() - now} seconds')


open_and_save_mnist()
