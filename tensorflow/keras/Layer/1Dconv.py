import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Reshape, Flatten, MaxPool1D

#test data
x = np.array([[1,2,3,4,5]], dtype= 'float64')
print(x.shape) #(1,5)

x=Reshape((5,1))(x) #
conv1d = Conv1D(filters=1,kernel_size=2,activation=None,use_bias=False)
x = conv1d(x)

print(conv1d.kernel)
print(x)

pool = MaxPool1D(pool_size=2)
pool_x = pool(x)
print(pool_x)

flatten = Flatten()(pool_x)
print(flatten)