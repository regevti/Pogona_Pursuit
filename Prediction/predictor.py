# Module responsible for real-time analysis and prediction of the pogona

from Detector.detector import Detector_v4


class Predictor:
    def __init__(self):
        self.detector = Detector_v4()
        self.frame_num = 0

    def handle_frame(frame):
        """
        Process a single frame, update prediction, and send prediction as
        an MQTT 
        """
