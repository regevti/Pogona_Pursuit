from datetime import datetime
import paho.mqtt.client as mqtt
from cache import RedisCache, CacheColumns as cc
from loggers import get_logger
from db_models import ORM


class Periphery:
    """class for communicating with reptilearn's arena.py"""
    def __init__(self):
        self.logger = get_logger('Periphery')
        self.cache = RedisCache()
        self.mqtt_client = mqtt.Client()
        self.orm = ORM()

    def mqtt_publish(self, topic, payload):
        self.mqtt_client.connect("localhost", 1883, 60)
        self.mqtt_client.publish(topic, payload)

    def feed(self):
        if self.cache.get(cc.IS_REWARD_TIMEOUT):
            return
        self.cache.set(cc.IS_REWARD_TIMEOUT, True)
        self.mqtt_publish('arena_command', '["dispense","Feeder"]')
        n_rewards_left = int(self.cache.get(cc.REWARD_LEFT))
        self.cache.set(cc.REWARD_LEFT, max(n_rewards_left - 1, 0))
        self.logger.info('Reward was given')
        self.orm.commit_reward(datetime.now())

