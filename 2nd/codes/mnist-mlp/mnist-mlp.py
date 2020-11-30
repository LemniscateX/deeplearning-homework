import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

figurey_loss = []
figurey_accuracy = []

learning_rate = 0.005
batch_size = 100
training_epochs = 4001

num_input = 784
num_hidden_1 = 256
num_hidden_2 = 256
num_output = 10

X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_output])

w1 = tf.Variable(tf.random_normal([num_input, num_hidden_1]))
w2 = tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2]))
w3 = tf.Variable(tf.random_normal([num_hidden_2, num_output]))

b1 = tf.Variable(tf.random_normal([num_hidden_1]))
b2 = tf.Variable(tf.random_normal([num_hidden_2]))
b3 = tf.Variable(tf.random_normal([num_output]))

layer1 = tf.nn.relu(tf.add(tf.matmul(X, w1), b1))
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, w2), b2))
logits = tf.add(tf.matmul(layer2, w3), b3)

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
    plt.savefig("mlp-loss.png")

    plt.figure()
    plt.plot(range(0, training_epochs, 50), figurey_accuracy)
    plt.title("Accuracy along the iterate times")
    plt.xlabel("Iterate times")
    plt.ylabel("Accuracy")
    plt.savefig("mlp-accuracy.png")

# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     for epoch in range(training_epochs):
#         avg_loss = 0.
#         avg_accuracy = 0.
#         total_batch = int(mnist.train.num_examples / batch_size)
#         for i in range(total_batch):
#             batch_data, batch_label = mnist.train.next_batch(batch_size)
#             _, l, a = sess.run([optimizer, loss, accuracy], feed_dict={X: batch_data, Y: batch_label})
#             avg_loss += l / total_batch
#             avg_accuracy += a / total_batch
#         print("Epoch:{0} loss={1} accuracy={2}".format(epoch, avg_loss, avg_accuracy))
#     finalLoss, finalAccuracy = sess.run([loss, accuracy], feed_dict={X: batch_data, Y: batch_label})
#     print("Optimization finished!")
#     print("Loss: {0} Accuracy: {1}".format(finalLoss, finalAccuracy))
