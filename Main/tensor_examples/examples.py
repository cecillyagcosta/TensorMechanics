import pandas as pd
import numpy as np
import seaborn as sns
import xarray as xr
import tensorflow as tf

print(tf.__version__)

#Graph Example

a = tf.constant(5)
b = tf.constant(3)
c = tf.constant(2)

d = tf.multiply(a, b)
e = tf.add(b, c)
f = tf.subtract(d, e)

#Numpay arrays as matrixes

a1 = tf.constant(np.array([[1, 0, 4], [2, 4, 3]]))
a2 = tf.constant(np.array([[0, 0, 5], [9, 2, 3]]))
a3 = tf.constant(np.array([[0,0], [1,0], [0,1]]))

# var.shape to show array's dimension --> # x lines, y columns 
# tf.matmul to multiplay matrixes

r1 = tf.matmul(a1, a3)

#Tranpose: 1 0 4  --> 1 2
#          2 4 3      0 4
#                     4 3

a4 = tf.transpose(a1)

#Using vars in tensorflow.
var = tf.Variable(3)

#Creating random matrixes

m = tf.random.normal((3,5), 0, 1) #(shape, average, standard deviation)

#Placeholders

data_x = np.random.randn(4, 8)
data_y = np.random.randn(8, 2)

b = tf.random.normal((4, 2), 0, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=(4,8))
y = tf.compat.v1.placeholder(tf.float32, shape=(8,2))

operation = tf.matmul(x, y) + b

max_value = tf.reduce_max(operation) #Return the max value of matrix 'operation'

with tf.compat.v1.Session() as session:
    out1 = session.run(operation, feed_dict={x: data_x, y: data_y})
    out2 = session.run(max_value, feed_dict={x: data_x, y: data_y})