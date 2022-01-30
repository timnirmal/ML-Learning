import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python import tf2
import keras


# Generate x and y data list for plotting
def generate_data(n_samples, noise=0.0):
    x = np.linspace(0, 1, n_samples)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, noise, n_samples)
    return x, y


if __name__ == '__main__':
    x, y = generate_data(100, noise=0.1)

    # figsize is in inches and dpi changes the resolution of the image
    # plt.figure(figsize=(2, 0.5), dpi=300)

    plt.plot(x, y, label='data', color='blue', linestyle='-', marker='.', linewidth=2, markersize=10, alpha=0.5,
             markerfacecolor='red')
    # Short Notation '[color][marker][line]' ex: 'b.--'
    plt.title('Sin(2πx)',
              fontdict={'fontsize': 20, 'fontweight': 'bold', 'fontname': 'Times New Roman', 'color': 'red'})

    plt.xlabel('xlabel')
    plt.ylabel('ylabel')

    plt.xticks([0, 0.25, 0.5, 0.75, 1, 1.25])
    plt.yticks([-1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25])

    # Adding Second Line
    x2 = np.arange(0, 1.2, 0.1)     # Intervels we want plot points

    plt.plot(x2[:6], x2[:6] ** 2, color="yellow", linewidth=2, label="y=x^2", marker='o')
    plt.plot(x2[5:], x2[5:] ** 2, color="yellow", linestyle="--", linewidth=2, label="y=x^2")

    plt.legend()  # plt.legend(['sin(2πx)'], loc='upper left')
    plt.show()
