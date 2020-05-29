import time
from threading import Thread

import gpustat


class GPUStatMonitor(Thread):
    def __init__(self, delay):
        super(GPUStatMonitor, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls to GPUtil

        self.reset()
        self.start()

    def get_updated_average_value(self, gpu_stat, average_stat, attribute_name):
        gpu_stat_value = getattr(gpu_stat, attribute_name)
        average_stat_value = getattr(average_stat, attribute_name)

        if gpu_stat_value is None or average_stat_value is None:
            return None

        return int(
            (average_stat_value * (self.total_number_of_entries - 1) + gpu_stat_value)
            / self.total_number_of_entries
        )

    def add_entry_to_average_stats(self, entry):
        self.total_number_of_entries += 1

        if self.total_number_of_entries == 1:
            # Initializing self.average_stats
            self.average_stats = entry
            return self.average_stats

        updated_stat_collection = []
        for gpu_index, gpu_stat in enumerate(entry):
            gpu_average_stats = self.average_stats[gpu_index]

            updated_stat_collection.append(
                gpustat.GPUStat(
                    {
                        "index": gpu_average_stats.index,
                        "uuid": gpu_average_stats.uuid,
                        "name": gpu_average_stats.name,
                        "memory.total": (gpu_average_stats.memory_total),
                        "memory.used": (
                            self.get_updated_average_value(
                                gpu_stat, gpu_average_stats, "memory_used"
                            )
                        ),
                        "memory_free": (
                            self.get_updated_average_value(
                                gpu_stat, gpu_average_stats, "memory_free"
                            )
                        ),
                        "memory_available": (
                            self.get_updated_average_value(
                                gpu_stat, gpu_average_stats, "memory_available"
                            )
                        ),
                        "temperature.gpu": (
                            self.get_updated_average_value(
                                gpu_stat, gpu_average_stats, "temperature"
                            )
                        ),
                        "fan.speed": (
                            self.get_updated_average_value(
                                gpu_stat, gpu_average_stats, "fan_speed"
                            )
                        ),
                        "utilization.gpu": (
                            self.get_updated_average_value(
                                gpu_stat, gpu_average_stats, "utilization"
                            )
                        ),
                        "power.draw": (
                            self.get_updated_average_value(
                                gpu_stat, gpu_average_stats, "power_draw"
                            )
                        ),
                        "enforced.power.limit": gpu_average_stats.power_limit,
                        "processes": gpu_average_stats.processes,
                    }
                )
            )

        self.average_stats = gpustat.GPUStatCollection(updated_stat_collection)

        return self.average_stats

    def run(self):
        """
        Runs `gpustat` query every `delay` to parse information from nvidia-smi
        until GPUStatMonitor.stop() is called.
        """
        while not self.stopped:
            entry = gpustat.GPUStatCollection.new_query()
            self.add_entry_to_average_stats(entry)

            time.sleep(self.delay)

    def stop(self):
        """Stop the recording of information from nvidia-smi runs"""
        self.stopped = True
        return self.average_stats

    def reset(self):
        """Reset the stats average"""
        self.average_stats = None
        self.total_number_of_entries = 0

    def display_average_stats_per_gpu(self):
        print(self.average_stats)


if __name__ == "__main__":
    # Instantiate monitor with a 10-second delay between updates
    monitor = GPUStatMonitor(1)

    # Do your action
    time.sleep(5)

    # Close monitor
    stats = monitor.stop()
