import numpy as np
import tensorflow as tf

from gpumonitor.callbacks.tf import TFGpuMonitorCallback

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1) / 255.
y_train = tf.keras.utils.to_categorical(y_train)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy")

model.fit(x_train, y_train, callbacks=[TFGpuMonitorCallback(0.5)])
