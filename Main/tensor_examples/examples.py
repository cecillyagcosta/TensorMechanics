import pandas as pd
import numpy as np
import seaborn as sns
import xarray as xr
import tensorflow as tf

#Graph Example
print(tf.__version__)

a = tf.constant(5)
b = tf.constant(3)
c = tf.constant(2)

d = tf.multiply(a, b)
e = tf.add(b, c)
f = tf.subtract(d, e)

print(f)

