#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf


# In[2]:


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0:1], 'GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  pass


# In[3]:


def multiple_gradients(x, y, point = 1, activation = tf.keras.activations.tanh, learning_rate = 0.01):
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    y = tf.convert_to_tensor(y, dtype=tf.float64)


    model = tf.keras.Sequential([
    layers.Dense(100, activation=activation, input_shape=[1,]),
    layers.Dropout(0.5), 
    layers.Dense(100, activation=activation),
    layers.Dropout(0.5),
    layers.Dense(100, activation=activation),
    layers.Dense(1)
    ])

    model.compile(
        run_eagerly=False,
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss=tf.keras.losses.mean_squared_error,
    )

    model.fit(x, y, epochs=20, batch_size=128)


    X = tf.convert_to_tensor([[point]], dtype=tf.float64)

    with tf.GradientTape() as tapef3:
        tapef3.watch(X)
        with tf.GradientTape(persistent = True) as tapef2:
            tapef2.watch(X)
            with tf.GradientTape(persistent = True) as tapef1:
                tapef1.watch(X)
                y = model(X)
#                 yprime = tapef1.gradient(y,X)
    #             print("predicted: ", y, "actual: ", tf.math.sin(X))
            grad1 = tapef1.gradient(y, X)
#             print("gradient1: ", grad1)
        grad2 = tapef2.gradient(grad1, X)
#         print("gradient2: ", grad2)
    grad3 = tapef3.gradient(grad2,X)
#     print("gradient3: ", grad3)

    return model(X).numpy(), grad1.numpy(), grad2.numpy(), grad3.numpy()

# print(multiple_gradients(x, y, point = 1, activation = tf.keras.activations.tanh, learning_rate = 0.01))


# In[4]:


def grad(x, y, point=1):
    x_min = np.min(x)
    x_max = np.max(x)
    x_len = len(x)
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    y = tf.convert_to_tensor(y, dtype=tf.float64)

    model = tf.keras.Sequential([
    layers.Dense(100, activation=tf.keras.activations.relu, input_shape=[1,]),
    layers.Dropout(0.5),   
    layers.Dense(100, activation=tf.keras.activations.relu),
    layers.Dropout(0.5),
    layers.Dense(100, activation=tf.keras.activations.relu),
    layers.Dense(1)
    ])

    # compile sets the training parameters
    model.compile(
        run_eagerly=False,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=tf.keras.losses.mean_squared_error,
    )

    model.fit(x, y, epochs=20, batch_size=128)

    X_point = tf.convert_to_tensor([[point]], dtype=tf.float64)
    X_all = tf.linspace(x_min, x_max, x_len, )

    with tf.GradientTape(persistent = True) as tapef1:
        tapef1.watch(X_all)
        y = model(X_all)
    grad_all = tapef1.gradient(y, X_all)

    with tf.GradientTape() as tape:
      tape.watch(X_point)
      y = model(X_point)
    grad_point = tape.gradient(y, X_point)

    return grad_point, grad_all, X_all, model(X_point)

def three_grads(x, y, point = 1):
    grad_point_a, grad_all_a, X_all_a, y_point = grad(x,y, point = point)
    grad_point_b, grad_all_b, X_all_b, _ = grad(X_all_a,grad_all_a)
    grad_point_c, grad_all_c, X_all_c, _ = grad(X_all_b,grad_all_b)
    return y_point.numpy(), grad_point_a.numpy(), grad_point_b.numpy(), grad_point_c.numpy()

# print(three_grads(x, y, 1))

