import tensorflow as tf

from gpumonitor.monitor import GPUStatMonitor


class TFGpuMonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self, delay=1, display_options=None):
        super(TFGpuMonitorCallback, self).__init__()
        self.delay = delay
        self.display_options = display_options if display_options else {}

    def on_epoch_begin(self, epoch, logs=None):
        self.monitor = GPUStatMonitor(self.delay, self.display_options)

    def on_epoch_end(self, epoch, logs=None):
        self.monitor.stop()
        print("")
        self.monitor.display_average_stats_per_gpu()
