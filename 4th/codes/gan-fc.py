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
training_epochs = 80001
learning_rate = 2e-4

gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
dis_input = tf.placeholder(tf.float32, shape=[None, img_dim])


def weight_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


w1_gen = tf.Variable(weight_init([noise_dim, gen_dim]))
w2_gen = tf.Variable(weight_init([gen_dim, img_dim]))
w1_dis = tf.Variable(weight_init([img_dim, dis_dim]))
w2_dis = tf.Variable(weight_init([dis_dim, 1]))

b1_gen = tf.Variable(tf.zeros([gen_dim]))
b2_gen = tf.Variable(tf.zeros([img_dim]))
b1_dis = tf.Variable(tf.zeros([dis_dim]))
b2_dis = tf.Variable(tf.zeros([1]))


def generator(x):
    hidden = tf.nn.relu(tf.add(tf.matmul(x, w1_gen), b1_gen))
    output = tf.nn.sigmoid(tf.add(tf.matmul(hidden, w2_gen), b2_gen))
    return output


def discriminator(x):
    hidden = tf.nn.relu(tf.add(tf.matmul(x, w1_dis), b1_dis))
    output = tf.nn.sigmoid(tf.add(tf.matmul(hidden, w2_dis), b2_dis))
    return output


gen_output = generator(gen_input)

dis_real_output = discriminator(dis_input)
dis_fake_output = discriminator(gen_output)

loss_gen = -tf.reduce_mean(tf.log(dis_fake_output))
loss_dis = -tf.reduce_mean(tf.log(dis_real_output) + tf.log(1. - dis_fake_output))

vars_gen = [w1_gen, w2_gen, b1_gen, b2_gen]
vars_dis = [w1_dis, w2_dis, b1_dis, b2_dis]

optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_gen, var_list=vars_gen)
optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_dis, var_list=vars_dis)

images = []
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for epoch in range(training_epochs):
        batch_images, _ = mnist.train.next_batch(batch_size)
        noise = np.random.uniform(-1, 1, size=[batch_size, noise_dim])
        feed_dict = {dis_input: batch_images, gen_input: noise}
        _, _, gl, dl = sess.run([optimizer_gen, optimizer_dis, loss_gen, loss_dis], feed_dict=feed_dict)
        if epoch % 2000 == 0:
            print("Epoch:{0} Generator Loss:{1} Discriminator Loss:{2}".format(epoch, gl, dl))
    finalLoss_gen, finalLoss_dis = sess.run([loss_gen, loss_dis], feed_dict=feed_dict)
    print("\nFinal: Generator Loss:{0} Discriminator Loss:{1}".format(finalLoss_gen, finalLoss_dis))

    for i in range(10):
        z = np.random.uniform(-1., 1., size=(1, noise_dim))
        g = sess.run(gen_output, feed_dict={gen_input: z})
        images.append(g)

for i in range(10):
    plt.imshow(images[i].reshape(28, 28), cmap='gray_r')
    plt.savefig('./gan-fc-outputs/{0}.png'.format(i))
