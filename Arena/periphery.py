import json
import time
from datetime import datetime
import paho.mqtt.client as mqtt
import paho.mqtt.subscribe as subscribe
from cache import RedisCache, CacheColumns as cc
from loggers import get_logger
from db_models import ORM
import config


class Periphery:
    """class for communicating with reptilearn's arena.py"""
    def __init__(self):
        self.logger = get_logger('Periphery')
        self.cache = RedisCache()
        self.mqtt_client = mqtt.Client()
        self.orm = ORM()

    def mqtt_publish(self, topic, payload):
        self.mqtt_client.connect(config.mqtt['host'], config.mqtt['port'], keepalive=60)
        self.mqtt_client.publish(topic, payload)

    def cam_trigger(self, state):
        assert state in [0, 1]
        self.mqtt_publish(config.mqtt['publish_topic'], f'["set","Camera Trigger",{state}]')

    def feed(self):
        if self.cache.get(cc.IS_REWARD_TIMEOUT):
            return
        self.cache.set(cc.IS_REWARD_TIMEOUT, True)
        self.mqtt_publish(config.mqtt['publish_topic'], '["dispense","Feeder"]')
        n_rewards_left = int(self.cache.get(cc.REWARD_LEFT))
        self.cache.set(cc.REWARD_LEFT, max(n_rewards_left - 1, 0))
        self.logger.info('Reward was given')
        self.orm.commit_reward(datetime.now())


class MQTTListener:
    topics = []

    def __init__(self, topics=None, callback=None, is_debug=True, stop_event=None, is_loop_forever=False):
        self.client = mqtt.Client()
        self.callback = callback
        self.topics = topics or self.topics
        self.topics = self.topics if isinstance(self.topics, (tuple, list)) else [self.topics]
        self.stop_event = stop_event
        self.is_debug = is_debug
        self.is_loop_forever = is_loop_forever
        self.is_initiated = False

    def init(self):
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(config.mqtt['host'], config.mqtt['port'])
        self.is_initiated = True

    def loop(self):
        if not self.is_initiated:
            self.init()
        if self.stop_event is not None:
            while not self.stop_event.is_set():
                self.client.loop()
        elif self.is_loop_forever:
            self.client.loop_forever()
        else:
            self.client.loop()

    def on_connect(self, client, userdata, flags, rc):
        if self.is_debug:
            print(f'MQTT connecting to host: {config.mqtt["host"]}; rc: {rc}')
        client.subscribe([(topic, 0) for topic in self.topics])

    def parse_payload(self, payload):
        return payload

    def on_message(self, client, userdata, msg):
        payload = self.parse_payload(msg.payload.decode('utf-8'))
        if self.callback is not None:
            self.callback(payload)
        if self.is_debug:
            print(f'received message with topic {msg.topic}: {payload}')


class TemperatureListener(MQTTListener):
    topics = ['arena/value']

    def parse_payload(self, payload):
        payload = json.loads(payload)
        return {k: v[0] for k, v in payload.items() if k in config.mqtt['temperature_sensors']}


if __name__ == "__main__":
    def hc_callback(payload):
        print(payload)

    listener = MQTTListener(topics=['healthcheck'], is_debug=False, callback=hc_callback)
    # listener.loop()
    while True:
        listener.loop()
        time.sleep(0.1)

