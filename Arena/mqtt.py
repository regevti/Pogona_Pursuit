import re
import json
import pandas as pd
from cache import CacheColumns, get_cache
from arena import UNSORTED_DIR
import paho.mqtt.client as mqtt

HOST = 'mosquitto'
SUBSCRIPTION_TOPICS = {
    'touch_log': 'event/log/touch',
    'score_log': 'event/log/score',
    'prediction_log': 'event/log/prediction'
}
TOUCHES_FILENAME = 'screen_touches.csv'


class MQTTClient:
    def __init__(self):
        self.client = mqtt.Client()
        self.cache = get_cache()

    def start(self):
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(HOST)
        # self.client.loop_forever()
        return self

    @staticmethod
    def on_connect(client, userdata, flags, rc):
        print(f'MQTT connecting to host: {HOST}; rc: {rc}')
        client.subscribe([(topic, 0) for topic in SUBSCRIPTION_TOPICS.values()])

    # @staticmethod
    def on_message(self, client, userdata, msg):
        payload = msg.payload.decode('utf-8')
        if is_match_topic(msg, 'touch_log'):
            data = json.loads(payload)
            df = pd.DataFrame([data])
            df.to_csv(self.get_csv_filename(), mode='a', header=False)
            print(f'saved to {self.get_csv_filename()}')

    def publish_event(self, topic, payload, retain=False):
        self.client.connect(HOST)
        self.client.publish(topic, payload, retain=retain)

    def publish_command(self, command, payload='', retain=False):
        self.publish_event(f'event/command/{command}', payload, retain)

    def get_csv_filename(self):
        if self.cache.get(CacheColumns.EXPERIMENT_NAME):
            parent = self.cache.get(CacheColumns.EXPERIMENT_PATH)
        else:
            parent = UNSORTED_DIR
        return f'{parent}/{TOUCHES_FILENAME}'


def is_match_topic(msg, topic_key):
    return re.match(SUBSCRIPTION_TOPICS[topic_key].replace('+', r'\w+'), msg.topic)




