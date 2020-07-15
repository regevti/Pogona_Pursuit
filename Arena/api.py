from flask import Flask, render_template, Response, request, make_response, send_file
import PySpin
import cv2
from arena import SpinCamera, Serializer, EXPOSURE_TIME, filter_cameras

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming ."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    vc = VideoStream()
    return Response(gen(vc),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


class VideoStream:
    def __init__(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        filter_cameras(self.cam_list, 'realtime')
        if len(self.cam_list) == 0:
            self.clear()
            raise Exception('No cameras were found')

        self.sc = SpinCamera(self.cam_list[0], None, None, is_stream=True)
        self.sc.begin_acquisition(EXPOSURE_TIME)
        self.serializer = Serializer()
        self.serializer.start_acquisition()

    def get_frame(self):
        image_result = self.sc.cam.GetNextImage()
        img = image_result.GetNDArray()
        return cv2.imencode(".jpg", img)

    def clear(self):
        self.cam_list.Clear()
        del self.sc
        self.system.ReleaseInstance()
        if getattr(self, 'serializer', None):
            self.serializer.stop_acquisition()

    def __del__(self):
        self.clear()


def gen(vc: VideoStream):
    """Video streaming generator function."""
    print('start web streaming')
    while True:
        (flag, encodedImage) = vc.get_frame()

        if not flag:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n\r\n')