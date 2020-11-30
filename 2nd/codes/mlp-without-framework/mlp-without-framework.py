from random import random, seed, shuffle
import matplotlib.pyplot as plt
import math

seed(10086)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


def mse(output, expected):
    return (expected - output) ** 2


def mse_derivative(output, expected):
    return expected - output


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.delta = None
        self.output = None

    def update(self, inputs, learning_rate):
        for i in range(len(inputs)):
            self.weights[i] += self.delta * inputs[i] * learning_rate
        self.bias += self.delta * learning_rate

    def feedforward(self, inputs):
        weighted_sum = 0.0
        for i in range(len(inputs)):
            weighted_sum += inputs[i] * self.weights[i]
        weighted_sum += self.bias
        self.output = sigmoid(weighted_sum)
        return self.output


class Layer:
    def __init__(self, neurons):
        self.neurons = neurons

    def update(self, inputs, learning_rate):
        for n in self.neurons:
            n.update(inputs, learning_rate)

    def feedforward(self, inputs):
        for n in self.neurons:
            n.feedforward(inputs)

    def outputs(self):
        output_list = []
        for n in self.neurons:
            output_list.append(n.output)
        return output_list


class Network:
    def __init__(self):
        self.layers = []

    def append_layer(self, in_dimension, out_dimension):
        neurons = []
        for _ in range(out_dimension):
            weights = []
            for _ in range(in_dimension):
                weights.append(random())
            bias = random()
            neuron = Neuron(weights, bias)
            neurons.append(neuron)
        layer = Layer(neurons)
        self.layers.append(layer)

    def update(self, inputs, learning_rate):
        for layer in self.layers:
            layer.update(inputs, learning_rate)
            inputs = layer.outputs()

    def feedforward(self, inputs):
        for layer in self.layers:
            layer.feedforward(inputs)
            inputs = layer.outputs()
        return inputs

    def feedback(self, expected):
        # 见PPT P31
        # 当j是输出单元时
        for i in range(len(expected)):
            rightmost_layer = self.layers[-1]
            neuron = rightmost_layer.neurons[i]
            neuron.delta = mse_derivative(neuron.output, expected[i])

        # 当j是隐层单元时单元时
        for i in range(len(self.layers) - 2, -1, -1):
            left_layer = self.layers[i]
            right_layer = self.layers[i + 1]
            for j in range(len(left_layer.neurons)):
                error = 0.0
                for rn in right_layer.neurons:
                    error += rn.weights[j] * rn.delta
                neuron = left_layer.neurons[j]
                neuron.delta = error * sigmoid_derivative(neuron.output)

    def get_loss(self, dataset):
        loss = 0.0
        for data in dataset:
            outputs = self.feedforward(data[0])
            expected = data[1]
            for i in range(len(expected)):
                loss += mse(outputs[i], expected[i])
        loss /= len(dataset)
        return loss


dataset = [[[0, 0], [0]],
           [[0, 1], [1]],
           [[1, 0], [1]],
           [[1, 1], [0]]]

network = Network()
network.append_layer(2, 2)
network.append_layer(2, 2)
network.append_layer(2, 1)

learning_rate = 0.3
training_epochs = 3001
figurey = []

for epoch in range(training_epochs):
    shuffle(dataset)
    for data in dataset:
        train_datum = data[0]
        train_label = data[1]
        network.feedforward(train_datum)
        network.feedback(train_label)
        network.update(train_datum, learning_rate)

    if epoch % 50 == 0:
        loss = network.get_loss(dataset)
        figurey.append(loss)
        print("Epoch: {0}, loss: {1}".format(epoch, loss))

loss = network.get_loss(dataset)
print("Loss: {}".format(loss))
for data in dataset:
    r = network.feedforward(data[0])
    print("expected: {0}, got: {1}".format(data[1][0], r[0]))

plt.figure()
plt.plot(range(0, training_epochs, 50), figurey)
plt.xlabel("Iterate times")
plt.ylabel("Loss along the iterate times")
plt.savefig("mlp-xor-loss.png")
