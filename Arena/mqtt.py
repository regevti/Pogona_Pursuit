import time
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from cache import CacheColumns, RedisCache
from parallel_port import ParallelPort
from utils import turn_display_off
import paho.mqtt.client as mqtt

HOST = os.environ.get('MQTT_HOST', 'mqtt')

LOG_TOPIC_PREFIX = 'event/log/'
LOG_TOPICS = {
    'touch': 'screen_touches.csv',
    'hit': 'hits.csv',
    'prediction': 'predictions.csv',
    'trajectory': 'bug_trajectory.csv'
}
SUBSCRIPTION_TOPICS = {
    'reward': 'event/command/reward',
    'led_light': 'event/command/led_light',
    'end_trial': 'event/command/end_trial',
    'end_experiment': 'event/command/end_experiment'
}
SUBSCRIPTION_TOPICS.update({k: LOG_TOPIC_PREFIX + k for k in LOG_TOPICS.keys()})
EXPERIMENT_LOG = 'event/log/experiment'


class MQTTClient:
    def __init__(self, parport: ParallelPort = None):
        self.client = mqtt.Client()
        self.cache = RedisCache()
        self.parport = parport
        self.live_manager = LiveExperimentManager(self, parport)

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

        elif msg.topic == SUBSCRIPTION_TOPICS['end_experiment']:
            self.live_manager.end_experiment()

        elif msg.topic == SUBSCRIPTION_TOPICS['end_trial']:
            self.live_manager.end_trial()

        elif msg.topic.startswith(LOG_TOPIC_PREFIX):
            topic = msg.topic.replace(LOG_TOPIC_PREFIX, '')
            try:
                payload = json.loads(payload)
                if topic == 'touch':
                    self.live_manager.handle_hit(payload)
                self.save_to_csv(topic, payload)
            except Exception as exc:
                print(f'Unable to parse log payload of {topic}: {exc}')

    def publish_event(self, topic, payload, retain=False):
        self.client.connect(HOST)
        self.client.publish(topic, payload, retain=retain)

    def publish_command(self, command, payload='', retain=False):
        self.publish_event(f'event/command/{command}', payload, retain)

    def save_to_csv(self, topic, payload):
        try:
            if topic == 'trajectory':
                df = pd.DataFrame(payload)
            else:
                df = pd.DataFrame([payload])
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


class MQTTPublisher(MQTTClient):
    @staticmethod
    def on_connect(client, userdata, flags, rc):
        pass


class LiveExperimentManager:
    def __init__(self, mqtt_client, parport):
        self.mqtt_client = mqtt_client
        self.cache = mqtt_client.cache
        self.parport = parport

    def log(self, msg):
        print(msg)
        self.mqtt_client.publish_event(EXPERIMENT_LOG, msg)

    def handle_hit(self, payload):
        if self.is_always_reward() and payload.get('is_hit') and payload.get('is_reward_bug'):
            self.cache.set(CacheColumns.EXPERIMENT_TRIAL_ON, False)  # stop trial
            return self.reward()

    def end_experiment(self):
        if self.cache.get(CacheColumns.EXPERIMENT_TRIAL_ON):
            self.end_trial()
        time.sleep(2)
        self.cache.delete(CacheColumns.EXPERIMENT_NAME)
        self.cache.delete(CacheColumns.EXPERIMENT_PATH)
        self.log('>> experiment finished\n')
        if self.is_always_reward:
            self.cache.delete(CacheColumns.ALWAYS_REWARD)

    def end_trial(self):
        self.mqtt_client.publish_command('hide_bugs')
        self.cache.delete(CacheColumns.EXPERIMENT_TRIAL_ON)
        turn_display_off()
        self.parport.led_lighting('off')

    def reward(self, is_force=False):
        if self.parport and (is_force or self.is_always_reward()):
            self.parport.feed()
            self.mqtt_client.publish_event(EXPERIMENT_LOG, '>> Reward was given')
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
