import keras
import numpy as np
import tensorflow as tf
import os
import math
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_model(w1,bias1, w2,bias2, x):
    flatten_num = 10 #10维数组
    layer = tf.reshape(x, [-1, flatten_num])
    layer = tf.matmul(layer, w1) + bias1
    layer = tf.nn.relu(tf.layers.batch_normalization(layer)+bias1)
    layer = tf.matmul(layer, w2) + bias2
    layer = tf.tanh(tf.layers.batch_normalization(layer)+bias2)
    layer = tf.nn.softmax(layer)
    return layer

def aiTest(images,shape):
    # 加载噪声
    images = (images/255-0.5)*2
    images = images + result
    images = (images*2 + 0.5)*255
    return images


# 加载数据
fashion_mnist = keras.datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
mnist = input_data.read_data_sets("../fashion_mnist", one_hot=True)
train_images = mnist.train.images
train_labels = mnist.train.labels
test_images = mnist.test.images
test_labels = mnist.test.labels
train_images = (train_images / 255.0 - 0.5) * 2
test_images = (test_images / 255.0 - 0.5) * 2
# t,占位符,10位数组,label
train_x = tf.placeholder(tf.float32, shape=(None, 10), name='x-input')
# x,占位符,28*28,image
train_y = tf.placeholder(tf.float32, shape=(None, 28, 28), name='y-input')
train_size = len(train_images)
test_size = len(test_images)
batch_size = 10
# 加载模型
w1_n = tf.Variable(np.load('w1_upset.npy'), trainable=False)
w2_n = tf.Variable(np.load('w2_upset.npy'), trainable=False)
bias1 = tf.Variable(np.load("bias1.npy"), trainable=False)
bias2 = tf.Variable(np.load("bias2.npy"), trainable=False)
x_label = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32)
x_label = tf.reshape(x_label, [-1, 10])  # 输入
model = get_model(w1_n, bias1, w2_n, bias2, x_label)
session = tf.Session()
init_op = tf.global_variables_initializer()
session.run(init_op)
# 生成噪声
result = session.run(model, feed_dict={
        train_x: np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).repeat(10, axis=0),
        train_y: np.reshape(test_images, (-1, 28, 28))})
result = (result + 1) / 2







