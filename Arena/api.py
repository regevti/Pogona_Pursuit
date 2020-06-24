from flask import Flask, render_template, Response, request, make_response, send_file
import PySpin
from arena import SpinCamera, Serializer, EXPOSURE_TIME

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming ."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


class VideoStream:
    def __init__(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        if len(self.cam_list) == 0:
            self.clear()
            raise Exception('No cameras were found')

        self.sc = SpinCamera(self.cam_list[0], None, None, is_stream=True)
        self.sc.begin_acquisition(EXPOSURE_TIME)
        self.serializer = Serializer()
        self.serializer.start_acquisition()

    def get_frame(self):
        self.sc.web_stream()

    def clear(self):
        self.cam_list.Clear()
        self.system.ReleaseInstance()
        if getattr(self, 'serializer', None):
            self.serializer.stop_acquisition()

    def __del__(self):
        self.clear()


def gen():
    """Video streaming generator function."""
    vc = VideoStream()
    try:
        vc.get_frame()
    except Exception:
        vc.clear()