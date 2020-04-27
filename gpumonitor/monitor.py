import time
from threading import Thread

import gpustat
import numpy as np


class GPUStatMonitor(Thread):
    def __init__(self, delay):
        super(GPUStatMonitor, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls to GPUtil
        self.stats = []
        self.start()

    def run(self):
        while not self.stopped:
            self.stats.append(gpustat.GPUStatCollection.new_query())
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        return self.stats

    def get_average_stats_per_gpu(self):
        is_stats_empty = len(self.stats) == 0
        if is_stats_empty:
            return None

        number_of_gpus = len(self.stats[0])
        stats_per_gpu = []
        for gpu_index in range(number_of_gpus):
            stats_per_gpu.append([stats[gpu_index] for stats in self.stats])

        return stats_per_gpu


    def display_average_stats_per_gpu(self):
        stats_per_gpu = self.get_average_stats_per_gpu()

        for stat in stats_per_gpu:
            tmp = gpustat.GPUStat({
                "index": stat[0].index,
                "uuid": stat[0].uuid,
                "name": stat[0].name,
                "memory.total": stat[0].memory_total,
                "memory.used": np.mean([element.memory_used for element in stat]),
                "memory_free": np.mean([element.memory_free for element in stat]),
                "memory_available": np.mean([element.memory_available for element in stat]),
                "temperature.gpu": np.mean([element.temperature for element in stat]),
                "fan.speed": np.mean([element.fan_speed for element in stat]),
                "utilization.gpu": np.mean([element.utilization for element in stat]),
                "power.draw": np.mean([element.power_draw for element in stat]),
                "enforced.power.limit": stat[0].power_limit,
                "processes": stat[0].processes,
            })
            print(tmp)
            pass


if __name__ == "__main__":
    # Instantiate monitor with a 10-second delay between updates
    monitor = GPUStatMonitor(1)

    # Do your action
    time.sleep(5)

    # Close monitor
    stats = monitor.stop()
