import re
import json
import pandas as pd
import paho.mqtt.client as mqtt

HOST = 'mosquitto'
SUBSCRIPTION_TOPICS = {
    'touch_log': 'event/log/touch'
}


class MQTTClient:
    def __init__(self, cache):
        self.client = mqtt.Client()
        self.cache = cache

    def start(self):
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(HOST)
        self.client.loop_forever()

    @staticmethod
    def on_connect(client, userdata, flags, rc):
        print(f'MQTT connecting to host: {HOST}; rc: {rc}')
        client.subscribe([(topic, 0) for topic in SUBSCRIPTION_TOPICS.values()])

    @staticmethod
    def on_message(client, userdata, msg):
        payload = msg.payload.decode('utf-8')
        if is_match_topic(msg, 'touch_log'):
            df = pd.read_json(payload)
            df.to_csv('my_csv.csv', mode='a', header=False)

    def publish_event(self, topic, payload, retain=False):
        self.client.publish(topic, payload, retain=retain)


def is_match_topic(msg, topic_key):
    return re.match(SUBSCRIPTION_TOPICS[topic_key].replace('+', r'\w+'), msg.topic)




