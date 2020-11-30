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

# Hypothesis
y_pred = tf.add(tf.multiply(X, W), b)

# Add stuffs
W_2 = tf.Variable(np.random.randn(), name="W2")
y_pred = tf.add(tf.multiply(tf.pow(X, 2), W_2), y_pred)
W_3 = tf.Variable(np.random.randn(), name="W3")
y_pred = tf.add(tf.multiply(tf.pow(X, 3), W_3), y_pred)
# Define hyperparameters
learning_rate = 0.01
training_epochs = 1000

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
                  "W1 =", sess.run(W),
                  "W2 =", sess.run(W_2),
                  "W3 =", sess.run(W_3),
                  "b =", sess.run(b))

    training_cost = sess.run(cost, feed_dict={X: x, Y: y})
    weight = sess.run(W)
    weight2 = sess.run(W_2)
    weight3 = sess.run(W_3)
    bias = sess.run(b)

    predictions = weight * x + weight2 * (x ** 2) + weight3 * (x ** 3) + bias
    print("Training cost =", training_cost,
          "Weight =", weight,
          "Weight2 =", weight2,
          "Weight3 =", weight3,
          "bias =", bias, '\n')

    plt.figure()
    plt.plot(x, y, 'ro', label='Original data')
    plt.plot(x, predictions, label='Fitted line')
    plt.title('Linear Regression Polynomial Result')
    plt.legend()
    plt.savefig("resultOfPoly.png")
