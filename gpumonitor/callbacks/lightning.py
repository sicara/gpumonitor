import pytorch_lightning as pl

from gpumonitor.monitor import GPUStatMonitor


class PyTorchGpuMonitorCallback(pl.callbacks.base.Callback):
    def __init__(self, delay):
        super(PyTorchGpuMonitorCallback, self).__init__()
        self.delay = delay

    def on_epoch_start(self, trainer, pl_module):
        self.monitor = GPUStatMonitor(self.delay)

    def on_epoch_end(self, trainer, pl_module):
        self.monitor.stop()
        print("")
        self.monitor.display_average_stats_per_gpu()
