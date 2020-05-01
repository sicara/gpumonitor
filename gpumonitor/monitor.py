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
        """
        Runs `gpustat` query every `delay` to parse information from nvidia-smi
        until GPUStatMonitor.stop() is called.
        """
        while not self.stopped:
            self.stats.append(gpustat.GPUStatCollection.new_query())
            time.sleep(self.delay)

    def stop(self):
        """Stop the recording of information from nvidia-smi runs"""
        self.stopped = True
        return self.stats

    def get_average_stats_per_gpu(self):
        """
        Build an average statistics of all nvidia-smi records.

        :return: a gpustat.GPUStatCollection with aggregated values
        """
        is_stats_empty = len(self.stats) == 0
        if is_stats_empty:
            return None

        number_of_gpus = len(self.stats[0])
        stats_per_gpu_over_time = []
        for gpu_index in range(number_of_gpus):
            stats_per_gpu_over_time.append([stats[gpu_index] for stats in self.stats])

        average_stats_per_gpu = gpustat.GPUStatCollection([
            gpustat.GPUStat({
                "index": gpu_stat_over_time[0].index,
                "uuid": gpu_stat_over_time[0].uuid,
                "name": gpu_stat_over_time[0].name,
                "memory.total": int(gpu_stat_over_time[0].memory_total),
                "memory.used": int(np.mean([element.memory_used for element in gpu_stat_over_time])),
                "memory_free": int(np.mean([element.memory_free for element in gpu_stat_over_time])),
                "memory_available": int(np.mean([element.memory_available for element in gpu_stat_over_time])),
                "temperature.gpu": int(np.mean([element.temperature for element in gpu_stat_over_time])),
                "fan.speed": np.mean([element.fan_speed for element in gpu_stat_over_time]),
                "utilization.gpu": np.mean([element.utilization for element in gpu_stat_over_time]),
                "power.draw": int(np.mean([element.power_draw for element in gpu_stat_over_time])),
                "enforced.power.limit": gpu_stat_over_time[0].power_limit,
                "processes": gpu_stat_over_time[0].processes,
            })
            for gpu_stat_over_time in stats_per_gpu_over_time
        ])

        return average_stats_per_gpu


    def display_average_stats_per_gpu(self):
        average_stats_per_gpu_collection = self.get_average_stats_per_gpu()
        print(average_stats_per_gpu_collection)



if __name__ == "__main__":
    # Instantiate monitor with a 10-second delay between updates
    monitor = GPUStatMonitor(1)

    # Do your action
    time.sleep(5)

    # Close monitor
    stats = monitor.stop()
