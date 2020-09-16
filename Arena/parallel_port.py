import time
import parallel


class ParallelPort:
    output = 0x00

    def __init__(self):
        self.p = parallel.Parallel()

    def turn_on(self):
        self.p.setData(self.p.getData() | self.output)

    def turn_off(self):
        self.p.setData(self.p.getData() & (self.output ^ 0xFF))

    def turn_on_for(self, duration=1):
        """Turn on for a certain duration in seconds"""
        self.turn_on()
        time.sleep(duration)
        self.turn_off()


class Feeder(ParallelPort):
    output = 0x01

    def feed(self):
        self.turn_on_for(2)
