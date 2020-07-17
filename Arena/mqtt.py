import paho.mqtt.client as mqtt

mqttc = mqtt.Client()


def publish_event(topic, payload):
    mqttc.connect('localhost')
    mqttc.publish(topic, payload)