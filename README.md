# gpumonitor

[![Pypi Version](https://img.shields.io/pypi/v/gpumonitor.svg)](https://pypi.org/project/gpumonitor/)
![Licence](https://img.shields.io/pypi/l/gpumonitor)
![Frameworks](https://img.shields.io/badge/Frameworks-PyTorchLightning%20|%20TensorFlow-blue.svg)


`gpumonitor` gives you **stats about GPU** usage during execution of your scripts and trainings,
as [TensorFlow](https://www.github.com/tensorflow/tensorflow) or 
[Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) callbacks.

<p align="center">
    <img src="./assets/callbacks.png" width="800" />
</p>


## Installation

Installation can be done directly from this repository:

```
pip install gpumonitor
```

## Getting started


<p align="center">
    <img src="./assets/gpumonitor.gif" width="1000" />
</p>

### Option 1: In your scripts

```python
monitor = gpumonitor.GPUStatMonitor(delay=1)

# Your instructions here
# [...]

monitor.stop()
monitor.display_average_stats_per_gpu()
```

It keeps track of the average of GPU statistics. To reset the average and start from fresh, you can also reset the monitor:

```python
monitor = gpumonitor.GPUStatMonitor(delay=1)

# Your instructions here
# [...]

monitor.display_average_stats_per_gpu()
monitor.reset()

# Some other instructions
# [...]

monitor.display_average_stats_per_gpu()
```

### Option 2: Callbacks

Add the following callback to your training loop:

For [TensorFlow](https://www.github.com/tensorflow/tensorflow),

```python
from gpumonitor.callbacks.tf import TFGpuMonitorCallback

model.fit(x, y, callbacks=[TFGpuMonitorCallback(delay=0.5)])
```

For [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning),

```python
from gpumonitor.callbacks.lightning import PyTorchGpuMonitorCallback

trainer = pl.Trainer(callbacks=[PyTorchGpuMonitorCallback(delay=0.5)])
trainer.fit(model)
```

## Display Format

You can customize the display format according to the `gpustat` options. For example, display of watts consumption,
fan speed are available. To know which options you can change, refer to:

- [TensorFlow callback example](https://github.com/sicara/gpumonitor/blob/42237f423254e8fc7ae21e8f2811533a4264064d/scripts/tf_training.py#L16)
- [`gpustat print_to()` docstring](https://github.com/wookayin/gpustat/blob/aba85f8eba9f7861022eb3dcc06ff771b451b3e1/gpustat/core.py#L178)

## Sources

- Built on top of [GPUStat](https://github.com/wookayin/gpustat)
- Separate thread loop coming from [gputil](https://github.com/anderskm/gputil)
