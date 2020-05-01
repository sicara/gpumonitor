# gpumonitor

gpumonitor gives you stats about GPU usage during execution of your scripts and trainings,
as TensorFlow (or PytorchLightning soon) callbacks.

<p align="center">
    <img src="./assets/callbacks.png" width="800" />
</p>


## Installation

Installation can be done directly from this repository:

```
pip install https://www.github.com/sicara/gpumonitor
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

### Option 2: Callbacks

Add the following callback to your training loop:

```python
callbacks = [gpumonitor.callbacks.TFGpuMonitorCallback(delay=1)]

model.fit(x, y, callbacks=callbacks)
```


## Sources

- Built on top of [GPUStat](https://github.com/wookayin/gpustat)
- Separate thread loop coming from [gputil](https://github.com/anderskm/gputil)
