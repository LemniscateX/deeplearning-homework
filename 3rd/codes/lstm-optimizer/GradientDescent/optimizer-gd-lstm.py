import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_observations = 600
x = np.linspace(1, num_observations, num_observations)
y = np.sin(0.06 * x) + np.random.uniform(-0.1, 0.1, num_observations)

train_input = []
train_output = []
test_input = []
test_output = []

for i in range(1, 414, 1):
    data = []
    for j in range(7):
        data.append(y[i + j - 1])
    train_input.append(data)
    train_output.append(y[i + 7 - 1])

train_input = np.reshape(train_input, (413, 7))
train_output = np.reshape(train_output, (413, 1))

for i in range(421, 594, 1):
    data = []
    for j in range(7):
        data.append(y[i + j - 1])
    test_input.append(data)
    test_output.append(y[i + 7 - 1])

test_input = np.reshape(test_input, (173, 7))
test_output = np.reshape(test_output, (173, 1))

learning_rate = 0.01
training_epochs = 501

X = tf.placeholder(tf.float32, [None, 7])
Y = tf.placeholder(tf.float32, [None, 1])

w = tf.Variable(tf.truncated_normal([32, 1], stddev=.1))
b = tf.Variable(tf.truncated_normal([1], stddev=.1))

cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=32)
_, last_state = tf.nn.dynamic_rnn(cell, tf.reshape(X, [-1, 7, 1]), dtype=tf.float32)
prediction = tf.nn.bias_add(tf.matmul(last_state.h, w), b)

loss = tf.reduce_mean(tf.pow(prediction - Y, 2))
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate).minimize(loss)

loss_list = []
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(training_epochs):
        batch = np.random.choice(train_input.shape[0], 100)
        sess.run(optimizer, feed_dict={X: train_input[batch], Y: train_output[batch]})
        if epoch % 10 == 0:
            l = sess.run(loss, feed_dict={X: train_input, Y: train_output})
            loss_list.append(l)
            print("Epoch:{0} loss={1}".format(epoch, l))

    test_prediction = sess.run(prediction, feed_dict={X: test_input})
    plt.figure()
    plt.legend((plt.scatter(x[427:], test_output),
                plt.scatter(x[427:], test_prediction)),
               ("Truth", "Prediction"))
    plt.savefig('optimizer-gd-lstm-difference.png')

plt.figure()
plt.plot(range(0, training_epochs, 10), loss_list)
plt.title("Loss along the iterate times")
plt.xlabel("Iterate times")
plt.ylabel("Loss")
plt.savefig("optimizer-gd-lstm-loss.png")
