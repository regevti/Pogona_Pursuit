import re
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from cache import CacheColumns, RedisCache
from parallel_port import Feeder
import paho.mqtt.client as mqtt

HOST = os.environ.get('MQTT_HOST', 'mqtt')
TOPIC_PREFIX = 'event/log/'
SUBSCRIPTION_TOPICS = {
    'touch': 'screen_touches.csv',
    'hit': 'hits.csv',
    'prediction': 'predictions.csv'
}

_feeder = None
try:
    _feeder = Feeder()
except Exception as exc:
    print(f'Error loading feeder: {exc}')


class MQTTClient:
    def __init__(self):
        self.client = mqtt.Client()
        self.cache = RedisCache()
        self.reward_manager = RewardManager(self.cache)

    def loop(self):
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(HOST)
        self.client.loop_forever()

    @staticmethod
    def on_connect(client, userdata, flags, rc):
        print(f'MQTT connecting to host: {HOST}; rc: {rc}')
        client.subscribe([(TOPIC_PREFIX + topic, 0) for topic in SUBSCRIPTION_TOPICS.keys()])

    def on_message(self, client, userdata, msg):
        payload = msg.payload.decode('utf-8')
        topic = msg.topic.replace(TOPIC_PREFIX, '')
        if topic in SUBSCRIPTION_TOPICS:
            if topic == 'hit':
                self.reward_manager.reward()
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

        return Path(f'{parent}/{SUBSCRIPTION_TOPICS[topic]}')


class RewardManager:
    def __init__(self, cache):
        self.cache = cache

    def reward(self, is_force=False):
        if _feeder and (is_force or self.is_reward_allowed()):
            _feeder.feed()
            return True

    def is_reward_allowed(self):
        return self.cache.get(CacheColumns.ALWAYS_REWARD)

# def is_match_topic(msg, topic_key):
#     return re.match(SUBSCRIPTION_TOPICS[topic_key].replace('+', r'\w+'), msg.topic)


if __name__ == '__main__':
    MQTTClient().loop()

