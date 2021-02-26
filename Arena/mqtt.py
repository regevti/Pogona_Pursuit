import json
from logger import save_to_csv, handle_hit, end_app_wait, end_experiment, reward, led_light, gaze_external, block_log
import paho.mqtt.client as mqtt
import config


class MQTTClient:
    def __init__(self):
        self.client = mqtt.Client()

    def loop(self):
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect('127.0.0.1')
        self.client.loop_forever()

    @staticmethod
    def on_connect(client, userdata, flags, rc):
        print(f'MQTT connecting to host: {config.mqtt_host}; rc: {rc}')
        client.subscribe([(topic, 0) for topic in config.subscription_topics.values()])

    def on_message(self, client, userdata, msg):
        payload = msg.payload.decode('utf-8')
        print(f'received message with topic {msg.topic}')
        if msg.topic == config.subscription_topics['reward']:
            reward.delay(is_force=True)

        elif msg.topic == config.subscription_topics['led_light']:
            led_light.delay(payload)

        elif msg.topic == config.subscription_topics['end_experiment']:
            end_experiment.delay()

        elif msg.topic == config.subscription_topics['gaze_external']:
            gaze_external.delay(payload)

        elif msg.topic == config.subscription_topics['end_app_wait']:
            end_app_wait.delay()

        elif msg.topic == config.subscription_topics['block_log']:
            block_log.delay(payload)

        elif msg.topic.startswith(config.log_topic_prefix):
            topic = msg.topic.replace(config.log_topic_prefix, '')
            try:
                payload = json.loads(payload)
                if topic == 'touch':
                    handle_hit.delay(payload)
                save_to_csv.delay(topic, payload)
            except Exception as exc:
                print(f'Unable to parse log payload of {topic}: {exc}')

    def publish_event(self, topic, payload, retain=False):
        self.client.connect(config.mqtt_host)
        self.client.publish(topic, payload, retain=retain)

    def publish_command(self, command, payload='', retain=False):
        self.publish_event(f'event/command/{command}', payload, retain)


class MQTTPublisher(MQTTClient):
    @staticmethod
    def on_connect(client, userdata, flags, rc):
        pass


if __name__ == '__main__':
    MQTTClient().loop()
