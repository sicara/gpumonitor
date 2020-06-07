import pytorch_lightning as pl

from gpumonitor.monitor import GPUStatMonitor


class PyTorchGpuMonitorCallback(pl.callbacks.base.Callback):
    def __init__(self, delay=1, display_options=None):
        super(PyTorchGpuMonitorCallback, self).__init__()
        self.delay = delay
        self.display_options = display_options if display_options else {}

    def on_epoch_start(self, trainer, pl_module):
        self.monitor = GPUStatMonitor(self.delay, self.display_options)

    def on_epoch_end(self, trainer, pl_module):
        self.monitor.stop()
        print("")
        self.monitor.display_average_stats_per_gpu()
