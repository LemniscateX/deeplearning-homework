import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

figurey_loss = []
figurey_accuracy = []

dropout_rate = 0.25
learning_rate = 0.01
batch_size = 100
training_epochs = 4001
padding = "SAME"

num_input = 784
num_output = 10

X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_output])

XR = tf.reshape(X, [-1, 28, 28, 1])
w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
w2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
w3 = tf.Variable(tf.truncated_normal([3136, 1024], stddev=0.1))
w4 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))

b1 = tf.Variable(tf.constant(0.1, shape=[32]))
b2 = tf.Variable(tf.constant(0.1, shape=[64]))
b3 = tf.Variable(tf.constant(0.1, shape=[1024]))
b4 = tf.Variable(tf.constant(0.1, shape=[10]))

c1 = tf.nn.conv2d(XR, w1, strides=[1, 1, 1, 1], padding=padding)
layer_1 = tf.nn.relu(tf.nn.bias_add(c1, b1))
p1 = tf.nn.max_pool(layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)

c2 = tf.nn.conv2d(p1, w2, strides=[1, 1, 1, 1], padding=padding)
layer_2 = tf.nn.relu(tf.nn.bias_add(c2, b2))
p2 = tf.nn.max_pool(layer_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)

layer_full_connection = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.reshape(p2, [-1, 3136]), w3), b3))
layer_dropout = tf.nn.dropout(layer_full_connection, dropout_rate)
logits = tf.nn.bias_add(tf.matmul(layer_dropout, w4), b4)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.math.argmax(logits, 1), tf.math.argmax(Y, 1)), tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(training_epochs):
        batch_data, batch_label = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={X: batch_data, Y: batch_label})
        if epoch % 50 == 0:
            l, a = sess.run([loss, accuracy], feed_dict={X: mnist.test.images, Y: mnist.test.labels})
            figurey_loss.append(l)
            figurey_accuracy.append(a)
            print("Epoch:{0} loss={1} accuracy={2}".format(epoch, l, a))
    finalLoss, finalAccuracy = sess.run([loss, accuracy], feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    print("\nloss={0} accuracy={1}".format(finalLoss, finalAccuracy))

    plt.figure()
    plt.plot(range(0, training_epochs, 50), figurey_loss)
    plt.title("Loss along the iterate times")
    plt.xlabel("Iterate times")
    plt.ylabel("Loss")
    plt.savefig("cnn-loss.png")

    plt.figure()
    plt.plot(range(0, training_epochs, 50), figurey_accuracy)
    plt.title("Accuracy along the iterate times")
    plt.xlabel("Iterate times")
    plt.ylabel("Accuracy")
    plt.savefig("cnn-accuracy.png")