import re
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from cache import CacheColumns, RedisCache
from parallel_port import ParallelPort
import paho.mqtt.client as mqtt

HOST = os.environ.get('MQTT_HOST', 'mqtt')

LOG_TOPIC_PREFIX = 'event/log/'
LOG_TOPICS = {
    'touch': 'screen_touches.csv',
    'hit': 'hits.csv',
    'prediction': 'predictions.csv'
}
SUBSCRIPTION_TOPICS = {
    'reward': 'event/command/reward',
    'led_light': 'event/command/led_light'
}
SUBSCRIPTION_TOPICS.update({k: LOG_TOPIC_PREFIX + k for k in LOG_TOPICS.keys()})


class MQTTClient:
    def __init__(self, parport: ParallelPort = None):
        self.client = mqtt.Client()
        self.cache = RedisCache()
        self.parport = parport
        self.live_manager = LiveExperimentManager(self.cache, parport)

    def loop(self):
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(HOST)
        self.client.loop_forever()

    @staticmethod
    def on_connect(client, userdata, flags, rc):
        print(f'MQTT connecting to host: {HOST}; rc: {rc}')
        client.subscribe([(topic, 0) for topic in SUBSCRIPTION_TOPICS.values()])

    def on_message(self, client, userdata, msg):
        payload = msg.payload.decode('utf-8')
        if msg.topic == SUBSCRIPTION_TOPICS['reward']:
            self.live_manager.reward(is_force=True)

        elif msg.topic == SUBSCRIPTION_TOPICS['led_light']:
            self.parport.led_lighting(payload)

        elif msg.topic.startswith(LOG_TOPIC_PREFIX):
            topic = msg.topic.replace(LOG_TOPIC_PREFIX, '')
            if topic == 'hit':
                self.live_manager.handle_hit()
            self.save_to_csv(topic, payload)

    def publish_event(self, topic, payload, retain=False):
        self.client.connect(HOST)
        self.client.publish(topic, payload, retain=retain)

    def publish_command(self, command, payload='', retain=False):
        self.publish_event(f'event/command/{command}', payload, retain)

    def save_to_csv(self, topic, payload):
        try:
            data = json.loads(payload)
            df = pd.DataFrame([data])
            df['timestamp'] = datetime.now()
            filename = self.get_csv_filename(topic)
            if filename.exists():
                df.to_csv(filename, mode='a', header=False)
            else:
                df.to_csv(filename)
            print(f'saved to {filename}')
        except Exception as exc:
            print(f'ERROR saving event to csv; {exc}')

    def get_csv_filename(self, topic) -> Path:
        if self.cache.get(CacheColumns.EXPERIMENT_NAME):
            parent = self.cache.get(CacheColumns.EXPERIMENT_TRIAL_PATH)
        else:
            parent = f'events/{datetime.today().strftime("%Y%m%d")}'
            Path(parent).mkdir(parents=True, exist_ok=True)

        return Path(f'{parent}/{LOG_TOPICS[topic]}')


class LiveExperimentManager:
    def __init__(self, cache, parport):
        self.cache = cache
        self.parport = parport

    def handle_hit(self):
        if self.is_always_reward():
            self.cache.set(CacheColumns.EXPERIMENT_TRIAL_ON, False)
            self.reward()

    def reward(self, is_force=False):
        if self.parport and (is_force or self.is_always_reward()):
            self.parport.feed()
            return True

    def is_always_reward(self):
        return self.cache.get(CacheColumns.ALWAYS_REWARD)

# def is_match_topic(msg, topic_key):
#     return re.match(SUBSCRIPTION_TOPICS[topic_key].replace('+', r'\w+'), msg.topic)


if __name__ == '__main__':
    _parport = None
    try:
        _parport = ParallelPort()
    except Exception as exc:
        print(f'Error loading feeder: {exc}')

    MQTTClient(_parport).loop()
