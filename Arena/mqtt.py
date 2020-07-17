import paho.mqtt.client as mqtt

mqttc = mqtt.Client()


def publish_event(topic, payload):
    mqttc.connect('mosquitto')
    mqttc.publish(topic, payload)