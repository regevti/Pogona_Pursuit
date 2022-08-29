import multiprocessing as mp
import torch
from arena import ArenaProcess


class StrikeDetector(ArenaProcess):
    def __init__(self, *args, shm=None):
        super(StrikeDetector, self).__init__(*args)
        self.shm = shm

    def __str__(self):
        return f'strike-detector-{self.cam_name}'

    def run(self):
        while not self.stop_signal.is_set():
            pass