import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

num_observations = 100
x = np.linspace(-3, 3, num_observations)
y = np.sin(x) + np.random.uniform(-0.5, 0.5, num_observations)
n = len(x)

# plt.scatter(x, y)
# plt.savefig('1.png')

# Define placeholders to feed our training examples into the optimizer
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Declare trainable Tensorflow variables for Weights and Bias initialied in random
W = tf.Variable(np.random.randn(), name="W")
b = tf.Variable(np.random.randn(), name="b")

# Define hyperparameters
learning_rate = 0.01
training_epochs = 1000

# Hypothesis
y_pred = tf.add(tf.multiply(X, W), b)

# Mean Squared Error as Cost Function
cost = tf.reduce_sum(tf.pow(y_pred - Y, 2)) / (2 * n)

# Gradient Descent Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Global Variables Initializer
init = tf.global_variables_initializer()

# Start!
with tf.Session() as sess:

    sess.run(init)

    for epoch in range(training_epochs):
        for (_x, _y) in zip(x, y):
            sess.run(optimizer, feed_dict={X: _x, Y: _y})

        if (epoch + 1) % 50 == 0:
            c = sess.run(cost, feed_dict={X: x, Y: y})
            print("Epoch", (epoch + 1), ": cost =", c,
                  "W =", sess.run(W),
                  "b =", sess.run(b))

    training_cost = sess.run(cost, feed_dict={X: x, Y: y})
    weight = sess.run(W)
    bias = sess.run(b)

    predictions = weight * x + bias
    print("Training cost =", training_cost,
          "Weight =", weight,
          "bias =", bias, '\n')

    plt.figure()
    plt.plot(x, y, 'ro', label='Original data')
    plt.plot(x, predictions, label='Fitted line')
    plt.title('Linear Regression Polynomial Result')
    plt.legend()
    plt.savefig("resultOfSingle.png")
