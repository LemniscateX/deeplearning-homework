import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

img_dim = 784
gen_dim = 256
dis_dim = 256

noise_dim = 100
batch_size = 128
training_epochs = 6001
learning_rate_gen = 2e-4
learning_rate_dis = 2e-4
padding = "SAME"

gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
dis_input = tf.placeholder(tf.float32, shape=[None, img_dim])


def weight_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


w1_gen = tf.Variable(weight_init([noise_dim, gen_dim]))
w2_gen = tf.Variable(weight_init([gen_dim, img_dim]))
w1_dis = tf.Variable(weight_init([5, 5, 1, 16]))
w2_dis = tf.Variable(weight_init([3136, 256]))
w3_dis = tf.Variable(weight_init([256, 1]))
# w1_dis = tf.Variable(tf.truncated_normal([5, 5, 1, 16], stddev=0.1))
# w2_dis = tf.Variable(tf.truncated_normal([3136, 256], stddev=0.1))
# w3_dis = tf.Variable(tf.truncated_normal([256, 1], stddev=0.1))

b1_gen = tf.Variable(tf.zeros([gen_dim]))
b2_gen = tf.Variable(tf.zeros([img_dim]))
b1_dis = tf.Variable(tf.zeros([16]))
b2_dis = tf.Variable(tf.zeros([256]))
b3_dis = tf.Variable(tf.zeros([1]))


def generator(x):
    hidden = tf.nn.relu(tf.add(tf.matmul(x, w1_gen), b1_gen))
    output = tf.nn.sigmoid(tf.add(tf.matmul(hidden, w2_gen), b2_gen))
    return output


def discriminator(x):
    convolution = tf.nn.conv2d(tf.reshape(x, [-1, 28, 28, 1]), w1_dis, strides=[1, 1, 1, 1], padding=padding)
    hidden = tf.nn.relu(tf.nn.bias_add(convolution, b1_dis))
    pooling = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding=padding)
    full_connection = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.reshape(pooling, [-1, 3136]), w2_dis), b2_dis))
    output = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(full_connection, w3_dis), b3_dis))
    return output


gen_output = generator(gen_input)

dis_real_output = discriminator(dis_input)
dis_fake_output = discriminator(gen_output)

loss_gen = -tf.reduce_mean(tf.log(dis_fake_output))
loss_dis = -tf.reduce_mean(tf.log(dis_real_output) + tf.log(1. - dis_fake_output))

vars_gen = [w1_gen, w2_gen, b1_gen, b2_gen]
vars_dis = [w1_dis, w2_dis, w3_dis, b1_dis, b2_dis, b3_dis]

optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate_gen).minimize(loss_gen, var_list=vars_gen)
optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate_dis).minimize(loss_dis, var_list=vars_dis)

images = []
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for epoch in range(training_epochs):
        batch_images, _ = mnist.train.next_batch(batch_size)
        noise = np.random.uniform(-1, 1, size=[batch_size, noise_dim])
        feed_dict = {dis_input: batch_images, gen_input: noise}
        _, _, gl, dl = sess.run([optimizer_gen, optimizer_dis, loss_gen, loss_dis], feed_dict=feed_dict)
        if epoch % 100 == 0:
            print("Epoch:{0} Generator Loss:{1} Discriminator Loss:{2}".format(epoch, gl, dl))
    finalLoss_gen, finalLoss_dis = sess.run([loss_gen, loss_dis], feed_dict=feed_dict)
    print("\nFinal: Generator Loss:{0} Discriminator Loss:{1}".format(finalLoss_gen, finalLoss_dis))

    for i in range(10):
        z = np.random.uniform(-1., 1., size=(1, noise_dim))
        g = sess.run(gen_output, feed_dict={gen_input: z})
        images.append(g)

for i in range(10):
    plt.imshow(images[i].reshape(28, 28), cmap='gray_r')
    plt.savefig('./gan-cnn-outputs/{0}.png'.format(i))
