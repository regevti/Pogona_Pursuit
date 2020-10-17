import os
import json
from logger import LOG_TOPICS, save_to_csv, handle_hit, end_trial, end_experiment, reward, led_light
from cache import RedisCache
import paho.mqtt.client as mqtt

HOST = os.environ.get('MQTT_HOST', 'mqtt')

LOG_TOPIC_PREFIX = 'event/log/'
SUBSCRIPTION_TOPICS = {
    'reward': 'event/command/reward',
    'led_light': 'event/command/led_light',
    'end_trial': 'event/command/end_trial',
    'end_experiment': 'event/command/end_experiment'
}
SUBSCRIPTION_TOPICS.update({k: LOG_TOPIC_PREFIX + k for k in LOG_TOPICS.keys()})


class MQTTClient:
    def __init__(self):
        self.client = mqtt.Client()
        self.cache = RedisCache()

    def loop(self):
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect('127.0.0.1')
        self.client.loop_forever()

    @staticmethod
    def on_connect(client, userdata, flags, rc):
        print(f'MQTT connecting to host: {HOST}; rc: {rc}')
        client.subscribe([(topic, 0) for topic in SUBSCRIPTION_TOPICS.values()])

    def on_message(self, client, userdata, msg):
        payload = msg.payload.decode('utf-8')
        print(f'received message with topic {msg.topic}')
        if msg.topic == SUBSCRIPTION_TOPICS['reward']:
            reward.delay(is_force=True)

        elif msg.topic == SUBSCRIPTION_TOPICS['led_light']:
            led_light.delay(payload)

        elif msg.topic == SUBSCRIPTION_TOPICS['end_experiment']:
            end_experiment.delay()

        elif msg.topic == SUBSCRIPTION_TOPICS['end_trial']:
            end_trial.delay()

        elif msg.topic.startswith(LOG_TOPIC_PREFIX):
            topic = msg.topic.replace(LOG_TOPIC_PREFIX, '')
            try:
                payload = json.loads(payload)
                if topic == 'touch':
                    handle_hit.delay(payload)
                save_to_csv.delay(topic, payload)
            except Exception as exc:
                print(f'Unable to parse log payload of {topic}: {exc}')

    def publish_event(self, topic, payload, retain=False):
        self.client.connect(HOST)
        self.client.publish(topic, payload, retain=retain)

    def publish_command(self, command, payload='', retain=False):
        self.publish_event(f'event/command/{command}', payload, retain)


class MQTTPublisher(MQTTClient):
    @staticmethod
    def on_connect(client, userdata, flags, rc):
        pass


if __name__ == '__main__':
    MQTTClient().loop()
