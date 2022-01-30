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


# Plot the data
def plot_data(x, y):
    plt.plot(x, y, 'o', label='data')
    plt.legend()
    plt.show()


# Build the model
def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, input_shape=(1,)))
    model.compile(loss='mse', optimizer='sgd')
    return model
    # Describe the model in Natural Language
    # First layer: 1 input node, 1 output node


# Model is a sequential model, so we can add layers to it The first layer is a dense layer, which means that it has a
# single input node and a single output node The input shape is a tuple of integers, which means that the input will
# be a 1-dimensional array of length 1 The output shape is a tuple of integers, which means that the output will be a
# 1-dimensional array of length 1 model compile is used to compile the model, which means that we are telling the
# model what loss function to use and what optimizer to use The loss function is the mean squared error (MSE) The
# optimizer is the stochastic gradient descent (SGD) The model is compiled with the loss function and optimizer,
# and the model is ready to be trained


# Train the model
def train_model(model, x, y, epochs=1000, verbose=0, batch_size=32):
    history = model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return history


# Plot the loss
def plot_loss(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Plot the model
def plot_model(model, x, y):
    y_pred = model.predict(x).flatten()
    plt.scatter(x, y, label='data')
    plt.plot(x, y_pred, 'r-', label='prediction')
    plt.legend()
    plt.show()


# Main function
def main():
    # Generate data
    x, y = generate_data(100, noise=0.1)
    # Plot data
    plot_data(x, y)
    # Build model
    model = build_model()
    # Train model
    history = train_model(model, x, y, epochs=1000, verbose=1, batch_size=32)
    # Plot loss
    plot_loss(history)
    # Plot model
    plot_model(model, x, y)


# Run main function
if __name__ == '__main__':
    main()

# References
# 1. https://keras.io/getting-started/sequential-model-guide/
# 2. https://keras.io/getting-started/functional-api-guide/
# 3. https://keras.io/getting-started/functional-api-guide/#sequential-model-guide
# 4. https://keras.io/getting-started/functional-api-guide/#functional-api-guide
