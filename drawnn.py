# imports
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
plt.style.use('ggplot')

from skimage.data import astronaut
from scipy.misc import imresize

# absolute distance
def distance(p1, p2):
    return tf.abs(p1 - p2)

# linear function
def linear(X, n_input, n_output, activation=None, scope=None):
    with tf.variable_scope(scope or "linear"):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=tf.constant_initializer())
        h = tf.matmul(X, W) + b
        if activation is not None:
            h = activation(h)
        return h

# if image exists, get image and resize
def get_image(filename):
    if os.path.isfile(filename):
        img = plt.imread(filename)
    else:
        img = astronaut()
    img = imresize(img, (64, 64))
    return img

# loop over the image, storing inputs and outputs (to learn from)
def train_image(img, xs, ys):
    for row_i in range(img.shape[0]):
        for col_i in range(img.shape[1]):
            xs.append([row_i, col_i])
            ys.append(img[row_i, col_i])    

# get appropriate image(s)
img = get_image("invalid") - get_image("invalid")
counter = 0
path = "images/"
for f in os.listdir(path):
    counter += 1
    curr = get_image(path + f)
    img = (counter-1)*(img/float(counter)) + (curr/float(counter))

#plt.imshow(img)

# positions in image
xs = []

# corresponding colours at positions
ys = []

# train
train_image(img, xs, ys)

xs = np.array(xs)
ys = np.array(ys)

# Normalizing input
xs = (xs - np.mean(xs)) / np.std(xs)

# shapes
xs.shape, ys.shape

X = tf.placeholder(tf.float32, shape=[None, 2], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 3], name='Y')

# neurons in each layer
n_neurons = [2, 64, 64, 64, 64, 64, 64, 3]

# make predictions
current_input = X
for layer_i in range(1, len(n_neurons)):
    current_input = linear(
        X=current_input,
        n_input=n_neurons[layer_i - 1],
        n_output=n_neurons[layer_i],
        activation=tf.nn.relu if (layer_i+1) < len(n_neurons) else None,
        scope='layer_' + str(layer_i))
Y_pred = current_input

# define cost and optimizer
cost = tf.reduce_mean(tf.reduce_sum(distance(Y_pred, Y), 1))

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# number of iterations and batch size
n_iterations = 500
batch_size = 50

# run tf session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    prev_training_cost = 0.0
    for it_i in range(n_iterations):
        idxs = np.random.permutation(range(len(xs)))
        n_batches = len(idxs) // batch_size
        for batch_i in range(n_batches):
            idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
            sess.run(optimizer, feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})

        training_cost = sess.run(cost, feed_dict={X: xs, Y: ys})
        print(it_i, training_cost)

        if (it_i + 1) % 20 == 0:
            ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)
            fig, ax = plt.subplots(1, 1)
            img = np.clip(ys_pred.reshape(img.shape), 0, 255).astype(np.uint8)
            plt.imshow(img)
            plt.draw()
            plt.pause(0.001)
            plt.close()
            #plt.show()
