import time
import parallel

# Outputs
feeder = 0x01
led_light = 0x02


class ParallelPort:
    def __init__(self):
        self.p = parallel.Parallel()
        self.p.setData(0x00)

    def turn_on(self, output):
        self.p.setData(self.p.getData() | output)

    def turn_off(self, output):
        self.p.setData(self.p.getData() & (output ^ 0xFF))

    def turn_on_for(self, output, duration=1):
        """Turn on for a certain duration in seconds"""
        self.turn_on(output)
        time.sleep(duration)
        self.turn_off(output)

    def feed(self):
        self.turn_on_for(feeder, 3)

    def led_lighting(self, state='off'):
        if state == 'on':
            self.turn_on(led_light)
        else:
            self.turn_off(led_light)
