
# coding: utf-8

# Simple GAM MNIST implementation with Tensorflow
# paper : https://arxiv.org/abs/1406.2661
# result : https://i.imgur.com/IQZXAnj.png

import os, time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
z_size = 100
train_epoch = 100
w_init = tf.contrib.layers.xavier_initializer()
training = True

def generator(z, reuse=False):
    with tf.variable_scope('generator', reuse=reuse), tf.device('/device:GPU:0'):
        
        layer = tf.layers.dense(z, 1000, activation=tf.nn.relu, kernel_initializer=w_init)
        #layer = tf.layers.batch_normalization(layer, training=training)
        layer = tf.layers.dense(layer, 1000, activation=tf.nn.relu, kernel_initializer=w_init)
        layer = tf.layers.batch_normalization(layer, training=training)
        layer = tf.layers.dense(layer, 784, activation=None, kernel_initializer=w_init)
        g_d = tf.nn.tanh(layer)
        return g_d

    
def discriminator( x, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse), tf.device('/device:GPU:0'):
        layer = tf.layers.dense(x, 784, activation=tf.nn.leaky_relu, kernel_initializer=w_init)
        layer = tf.layers.dense(layer, 256, activation=tf.nn.leaky_relu, kernel_initializer=w_init)
        layer = tf.layers.dense(layer, 64, activation=tf.nn.leaky_relu, kernel_initializer=w_init)
        logits = tf.layers.dense(layer, 1, activation=None, kernel_initializer=w_init)
        d = tf.nn.sigmoid(logits)
        return d, logits

z = tf.random_uniform(shape=(batch_size, z_size), dtype=tf.float32)
x = tf.placeholder(shape=[batch_size, 784], dtype=tf.float32) # for mnist
x_fake = generator(z, reuse=False)

y_fake, logits_fake = discriminator(x_fake, False)
y_real, logits_real = discriminator(x, True)


# use logits and sigmoid_cross_entropy
# to remove sigmoid from backpropagation process
label_one = tf.ones_like(logits_real)
label_zero = tf.zeros_like(logits_fake)
loss_d = tf.losses.sigmoid_cross_entropy(multi_class_labels=label_zero, logits=logits_fake) + tf.losses.sigmoid_cross_entropy(multi_class_labels=label_one, logits=logits_real)
loss_g = tf.losses.sigmoid_cross_entropy(multi_class_labels=label_one, logits=logits_fake) + tf.losses.sigmoid_cross_entropy(multi_class_labels=label_zero, logits=logits_real)


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):

    d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
    g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]

    # optimize D
    d_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
    d_train = d_opt.minimize(loss_d, var_list=d_vars)

    # optimize G
    g_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
    g_train = d_opt.minimize(loss_g, var_list=g_vars)


# for Tensorboard
fake_images = tf.reshape(x_fake, [-1, 28, 28, 1])
tf.summary.image('fake_images', fake_images, 3)
tf.summary.histogram('y_real', y_real)
tf.summary.histogram('y_fake', y_fake)
tf.summary.scalar('loss_d', loss_d)
tf.summary.scalar('loss_g', loss_g)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1~1 from 0~1
train_label = mnist.train.labels


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
writer = tf.summary.FileWriter('./gan_mnist_ffnn', sess.graph)
merged = tf.summary.merge_all()


np.random.seed(int(time.time()))
start_time = time.time()
print('training start at {}'.format(start_time))

for epoch in range(train_epoch):
    
    # update discriminator
    for i in range(len(train_set) // batch_size):
        
        x_data = train_set[i * batch_size:(i + 1) * batch_size]

        _ = sess.run([d_train], {x: x_data})
        _ = sess.run([g_train], {x: x_data})
    
    if epoch % 10 == 0:
        _summ = sess.run(merged, {x: x_data})
        writer.add_summary(_summ, epoch)
        
        _fake_images = sess.run(fake_images, {x: x_data})
        
        plt.figure(figsize=(9,1))    
        for k in range(1, 8, 1):
            imgIdx = k
            graphIdx = ''.join(str(x) for x in [1,8,k])
            plt.subplot(graphIdx)
            plt.axis('off')
            plt.imshow(np.reshape(_fake_images[imgIdx], [28,28]), cmap="gray")
            
        plt.title("epoch {}".format(epoch))
        plt.show()
