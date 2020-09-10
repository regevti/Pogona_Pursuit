import time
import parallel


class ParallelPort:
    def __init__(self):
        self.p = parallel.Parallel()

    def turn_off(self):
        self.p.setData(0x00)


class Feeder(ParallelPort):
    def feed(self):
        self.p.setData(0x01)
        time.sleep(2)
        self.turn_off()
