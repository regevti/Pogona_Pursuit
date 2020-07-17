from flask import Flask, render_template, Response, request, make_response, send_file
import PySpin
import cv2
from mqtt import publish_event
from arena import SpinCamera, record, filter_cameras, \
    CAMERA_NAMES, DEFAULT_NUM_FRAMES, EXPOSURE_TIME

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming ."""
    return render_template('index.html', cameras=CAMERA_NAMES.keys(), exposure=EXPOSURE_TIME,
                           num_frames=DEFAULT_NUM_FRAMES)


@app.route('/record', methods=['POST'])
def record_video():
    """Record video"""
    if request.method == 'POST':
        data = request.json
        return Response(record(**data))


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    vc = VideoStream()
    return Response(gen(vc),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/init_bugs')
def init_bugs():
    publish_event('event/command/init_bugs', 1)


@app.route('/hide_bugs')
def hide_bugs():
    publish_event('event/command/hide_bugs', '')


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
        # self.serializer = Serializer()
        # self.serializer.start_acquisition()

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