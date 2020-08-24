import serial
import time

SERIAL_PORT = '/dev/ttyACM0'
SERIAL_BAUD = 9600


class Serializer:
    def __init__(self):
        self.ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(0.5)

    def start_acquisition(self):
        self.ser.write(b'H')

    def stop_acquisition(self):
        self.ser.write(b'L')
