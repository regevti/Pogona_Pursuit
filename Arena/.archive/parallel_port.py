import time
import parallel

# Outputs
feeder = 0x01
led_light = 0x04
heat_light = 0x08


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
            # self.turn_on(heat_light)
        else:
            self.turn_off(led_light)
            # self.turn_off(heat_light)

    def heat_lighting(self, state='off'):
        """notice for heat that the connections are opposite"""
        if state == 'on':
            self.turn_off(heat_light)
        else:
            self.turn_on(heat_light)



class ArenaOperations(Subscriber):
    sub_name = 'arena_operations'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the parallel port
        self.parport = None
        self.last_reward_time = None
        self.orm = ORM()
        if config.is_use_parport:
            try:
                self.parport = ParallelPort()
                self.logger.debug('Parallel port is ready')
            except Exception as exc:
                self.logger.exception(f'Error loading feeder: {exc}')
        else:
            self.logger.warning('Parallel port is not configured. some arena operation cannot work')

    def _run(self, channel, data):
        if self.parport is None:
            return
        channel = channel.split('/')[-1]
        if channel == 'reward':
            ret = self.reward()
            if ret:
                self.logger.info('Manual reward was given')
            else:
                self.logger.warning('Could not give manual reward')

        elif channel == 'led_light':
            if self.parport:
                self.logger.debug(f'LED lights turned {data}')
                self.parport.led_lighting(data)

        elif channel == 'heat_light':
            if self.parport:
                self.logger.debug(f'Heat lights turned {data}')
                self.parport.heat_lighting(data)

    def reward(self):
        if self.parport and not self.cache.get(cc.IS_REWARD_TIMEOUT):
            self.parport.feed()
            self.last_reward_time = time.time()
            self.cache.set(cc.IS_REWARD_TIMEOUT, True)
            self.cache.publish('cmd/visual_app/reward_given')
            self.orm.commit_reward(datetime.now())
            return True
