"""""""""""""""""""""
Environment Setting:

bash
pip install --upgrade virtualenv
virtualenv -p python3
brew install pyenv
pyenv install python3.5.6
pyenv global 3.5.6
"""""""""""""""""""""

# Test Tensorflow installed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))


"""""""""
" NUMPY "
"""""""""
import numpy as np 

np.load('data.npz')
data = np.arange(6) # array([0, 1, 2, 3, 4, 5])
data.reshape((3, 2))

np.zeros((3, 28, 28)).reshape((3, -1)) # (10, 784)


""""""""""""""
" TENSORFLOW "
""""""""""""""
import tensorflow as tf

### VARIABLE ###
with tf.name_scope():

with tf.variable_scope("encoder", reuse=reuse) as scope:
    x = tf.placeholder(tf.float32, [None, features]) # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.float32, [None, output]) # 0-9 digits recognition => 10 classes 
    W = tf.Variable(tf.zeros([features, output]), dtype=tf.float32)
    b = tf.Variable(tf.zeros([output]), dtype=tf.float32)

    v1 = tf.get_variable("var1", [1], dtype=tf.float32)
    v2 = tf.Variable(1, name="var2", dtype=tf.float32)
    a = tf.add(v1, v2)

    z = tf.Variable(tf.zeros(W.get_shape()))
    # TensorShape([Dimension(100), Dimension(784), Dimension(10), Dimension(1)])  
    print(W.get_shape())
    # [100, 784, 10, 1]
    print(W.get_shape().as_list())
    print(tf.shape(W))

### ARITHMITIC ###
tf.add(a, b)
tf.add(a, b)
tf.matmul(A, B)
tf.log(a)
tf.reduce_sum(tensor, axis=None)
tf.reduce_mean(tensor, axis=None)
tf.random_normal(shape, mean=0, stddev=1)
tf.truncated_normal(shape)
tf.argmax(logits, axis=1)
tf.constant(0, shape=shape)
tf.equal(tf.constant(0), tf.constant(1))
tf.less(tensor, 10) # tensor < 10
correct_prediction = tf.equal(tf.argmax(y_prob, 1), tf.argmax(y, 1))

### LAYER ###
input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
bn = tf.nn.batch_normalization(input_layer)

conv1 = tf.layers.conv2d(
  inputs=input_layer,
  filters=32,
  kernel_size=[5, 5],
  padding="same",
  activation=tf.nn.relu)

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# Dense
pool1_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

# Activation
tf.nn.sigmoid
tf.nn.tanh
tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h3']), biases['decoder_b3']))

# Logits Layer
logits = tf.layers.dense(inputs=dropout, units=10)

### LOSS ###
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
optimizer = tf.train.AdamOptimizer()

# z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
# = max(x, 0) - x * z + log(1 + exp(-abs(x)))
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
# Computes sigmoid of x element-wise.
# Specifically, y = 1 / (1 + exp(-x)).
loss = tf.nn.sigmoid(logis)
# softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
loss = tf.nn.softmax(logis)

train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

### SESSION ###
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run([train_op], 
            feed_dict={x: np.random.normal((batch_size, 28, 28, 1)})

# Interactive Session
sess = tf.InteractiveSession()
# training
# testing
sess.close()

### GPU MANAGEMENT ###
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

### Bug report ###
* cannot load mnist
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
run:
/Applications/Python 3.6/Install Certificates.command

""""""""""""
" APPENDIZ "
""""""""""""
# CONVOLUTION NUERAL NETWORK

# Padding
The general rule now, if a matrix NxN is convolved with fxf filter/kernel and padding p give us n+2p-f+1,n+2p-f+1 matrix.
