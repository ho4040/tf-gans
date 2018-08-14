# coding: utf-8

# DC Generative adversarial networks
# paper : https://arxiv.org/abs/1511.06434
# result : https://i.imgur.com/172koYy.png

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

def generator(z, reuse=False, color_channel=1):
    with tf.variable_scope('generator', reuse=reuse), tf.device('/device:GPU:0'):
         
        s_size = 4
        
        initializer = tf.random_normal_initializer(mean=0, stddev=0.02)

        z = tf.convert_to_tensor(z)
        with tf.variable_scope('reshape'): # z vector convert into tensor
            outputs = tf.layers.dense(z, 1024 * s_size * s_size)
            outputs = tf.reshape(outputs, [-1, s_size, s_size, 1024])
            outputs = tf.layers.batch_normalization(outputs, training=training)
            outputs = tf.nn.relu(outputs, name='outputs') # shape (batch_size, 4, 4, 1024)

        with tf.variable_scope('tconv1'): # Transposed conv 1
            tconv1 = tf.layers.conv2d_transpose(outputs, filters=512, kernel_size=[5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            tconv1 = tf.layers.batch_normalization(tconv1, training=training)
            tconv1 = tf.nn.relu(tconv1) # shape = (batch_size, 8, 8, 512)

        with tf.variable_scope('tconv2'): # Transposed conv 2
            tconv2 = tf.layers.conv2d_transpose(tconv1, filters=256, kernel_size=[5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            tconv2 = tf.layers.batch_normalization(tconv2, training=training)
            tconv2 = tf.nn.relu(tconv2) # output shape = (batch_size, 16, 16, 256)

        with tf.variable_scope('tconv3'): # Transposed conv 3
            tconv3 = tf.layers.conv2d_transpose(tconv2, filters=128, kernel_size=[5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            tconv3 = tf.layers.batch_normalization(tconv3, training=training)
            tconv3 = tf.nn.relu(tconv3) # output shape = (batch_size, 32, 32, 128)

        with tf.variable_scope('tconv4'): # Transposed conv 4 Filters == RGB
            tconv4 = tf.layers.conv2d_transpose(inputs=tconv3, filters=color_channel, kernel_size=[5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            #tconv4 = tf.layers.batch_normalization(tconv4, training=training)

        # tanh output
        g = tf.nn.tanh(tconv4) # output shape = (batch_size, 64, 64, 3)
        #g = tf.nn.sigmoid(tconv4)
        return g

    
def discriminator( x, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse), tf.device('/device:GPU:0'):
        
        initializer = tf.random_normal_initializer(mean=0, stddev=0.02)
        d_inputs = tf.convert_to_tensor(x)

        with tf.variable_scope('conv1'): # conv 1
            conv1 = tf.layers.conv2d(d_inputs, 64, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            conv1 = tf.layers.batch_normalization(conv1, training=training)
            conv1 = tf.nn.leaky_relu(conv1) # shape (batch_size, 32, 32, 64)

        with tf.variable_scope('conv2'): # conv 2
            conv2 = tf.layers.conv2d(conv1, 128, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            conv2 = tf.layers.batch_normalization(conv2, training=training)
            conv2 = tf.nn.leaky_relu(conv2) # shape (batch_size, 16, 16, 128)

        with tf.variable_scope('conv3'): # conv 3
            conv3 = tf.layers.conv2d(conv2, 256, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            conv3 = tf.layers.batch_normalization(conv3, training=training)
            conv3 = tf.nn.leaky_relu(conv3) # shape (batch_size, 4, 4, 256)

        # conv 4
        with tf.variable_scope('conv4'):
            conv4 = tf.layers.conv2d(conv3, 512, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            conv4 = tf.layers.batch_normalization(conv4, training=training)
            conv4 = tf.nn.leaky_relu(conv4) # shape (batch_size, 2, 2, 512)

        batch_size = conv4.get_shape()[0].value
        reshaped = tf.reshape(conv4, [batch_size, -1])
        logits = tf.layers.dense(reshaped, 1, name='d', activation=None) # TODO : remove dense
        d = tf.nn.sigmoid(logits)
        return d, logits



z = tf.random_normal(shape=(batch_size, z_size), mean=0.0, stddev=1.0, dtype=tf.float32)
#z = tf.random_uniform(shape=(batch_size, z_size), dtype=tf.float32)
x = tf.placeholder(shape=[batch_size, 784], dtype=tf.float32) # for mnist
reshaped = tf.reshape(x, (batch_size, 28, 28, 1)) # convert raw data to valid image data
resized = tf.image.resize_images(reshaped, (64,64,)) # change size as D input  
x_fake = generator(z, reuse=False)

y_fake, logits_fake = discriminator(x_fake, False)
y_real, logits_real = discriminator(resized, True)

#loss_d = -(tf.reduce_mean(tf.log(y_real)) + tf.reduce_mean(tf.log(1-y_fake)))
#loss_g = tf.reduce_mean(tf.log(1-y_fake))

# To remove sigmoid from backpropagation process, use logit and sigmoid_cross_entropy
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



fake_images = tf.reshape(x_fake, [-1, 64, 64, 1])
#tf.summary.image('fake_images', fake_images, 3)
#tf.summary.histogram('y_real', y_real)
#tf.summary.histogram('y_fake', y_fake)
#tf.summary.scalar('loss_d', loss_d)
#tf.summary.scalar('loss_g', loss_g)



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1~1 from 0~1
train_label = mnist.train.labels



sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#writer = tf.summary.FileWriter('./dcgan_mnist', sess.graph)
#merged = tf.summary.merge_all()
print("ready!")



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
        #_summ = sess.run(merged, {x: x_data})
        #writer.add_summary(_summ, epoch)
        
        _fake_images = sess.run(fake_images, {x: x_data})
        
        plt.figure(figsize=(9,1))    
        for k in range(1, 8, 1):
            imgIdx = k
            graphIdx = ''.join(str(x) for x in [1,8,k])
            plt.subplot(graphIdx)
            plt.axis('off')
            plt.imshow(np.reshape(_fake_images[imgIdx], [64,64]), cmap="gray")
            
        plt.title("epoch {}".format(epoch))
        plt.show()
        
        
