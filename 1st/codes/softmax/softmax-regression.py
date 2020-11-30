import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

num_features = 784
num_labels = 10
learning_rate = 0.05
batch_size = 128
num_steps = 5001

figureItemNum = num_steps // 50 + 1
figurex = np.arange(figureItemNum)
for j in range(figureItemNum):
    figurex[j] = figurex[j] * 50
figurey1 = np.zeros((figureItemNum, 1))
figurey2 = np.zeros((figureItemNum, 1))

train_dataset = mnist.train.images
train_labels  = mnist.train.labels
test_dataset  = mnist.test.images
test_labels   = mnist.test.labels
valid_dataset = mnist.validation.images
valid_labels  = mnist.validation.labels

graph = tf.Graph()

with graph.as_default():
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, num_features))
    tf_train_labels = tf.placeholder(
        tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    weights = tf.Variable(tf.truncated_normal([num_features, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=tf_train_labels, logits=logits))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(
        tf.matmul(tf_test_dataset, weights) + biases)


def accuracy(predictions, labels):
    correctly_predicted = np.sum(
        np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    
    i = 0
    for step in range(num_steps):
        offset = np.random.randint(0, train_labels.shape[0] - batch_size - 1)

        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        feed_dict = {tf_train_dataset: batch_data,
                     tf_train_labels: batch_labels}

        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        maccuracy = accuracy(predictions, batch_labels)
        vaccuracy = accuracy(valid_prediction.eval(), valid_labels)

        if (step % 500 == 0):
            print("Minibatch loss at step {0}: {1}".format(step, l))
            print("Minibatch accuracy: {:.1f}%".format(maccuracy))
            print("Validation accuracy: {:.1f}%\n".format(vaccuracy))

        if (step % 50 == 0):
            figurey1[i, 0] = l
            figurey2[i, 0] = round(vaccuracy / 100, 3)
            i = i + 1

    finalLoss = session.run(loss, feed_dict=feed_dict)
    finalAccuracy = accuracy(test_prediction.eval(), test_labels)

    print("\nLoss: {}".format(finalLoss))
    print("Test accuracy: {:.1f}%".format(finalAccuracy))

    plt.figure()
    plt.plot(figurex, figurey1)
    plt.title("Loss along the iterate times")
    plt.xlabel("Iterate times")
    plt.ylabel("Loss")
    plt.savefig("loss.png")

    plt.figure()
    plt.plot(figurex, figurey2)
    plt.title("Accuracy along the iterate times")
    plt.xlabel("Iterate times")
    plt.ylabel("Accuracy")
    plt.savefig("accuracy.png")