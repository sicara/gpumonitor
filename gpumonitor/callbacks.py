import numpy as np
import tensorflow as tf

from gpumonitor.monitor import GPUStatMonitor


class TFGpuMonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self, delay):
        super(TFGpuMonitorCallback, self).__init__()
        self.delay = delay

    def on_epoch_begin(self, epoch, logs=None):
        self.monitor = GPUStatMonitor(self.delay)

    def on_epoch_end(self, epoch, logs=None):
        self.monitor.stop()
        print("")
        self.monitor.display_average_stats_per_gpu()


if __name__ == "__main__":
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1) / 255.
    y_train = tf.keras.utils.to_categorical(y_train)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    model.fit(x_train, y_train, callbacks=[TFGpuMonitorCallback(0.5)])


