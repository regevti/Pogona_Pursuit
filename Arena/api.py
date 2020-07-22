from flask import Flask, render_template, Response, request, make_response, send_file
from cache import get_cache, CacheColumns
import PySpin
import cv2
from mqtt import MQTTClient
from utils import get_datetime_string, titlize, mkdir
from arena import SpinCamera, record, filter_cameras, \
    CAMERA_NAMES, EXPOSURE_TIME, ACQUIRE_STOP_OPTIONS, OUTPUT_DIR


app = Flask(__name__)
cache = get_cache(app)
mqtt_client = MQTTClient(cache).start()


@app.route('/')
def index():
    """Video streaming ."""
    return render_template('index.html', cameras=CAMERA_NAMES.keys(), exposure=EXPOSURE_TIME,
                           acquire_stop={k: titlize(k) for k in ACQUIRE_STOP_OPTIONS.keys()})


@app.route('/record', methods=['POST'])
def record_video():
    """Record video"""
    if request.method == 'POST':
        data = request.json
        data['cache'] = cache
        return Response(record(**data))


@app.route('/start_experiment', methods=['POST'])
def start_experiment():
    """Set Experiment Name"""
    if request.method == 'POST':
        data = request.json
        experiment_name = f'{data.get("name")}_{get_datetime_string()}'
        timeout = data.get('time')
        mkdir(f'{OUTPUT_DIR}/{experiment_name}')
        cache.set(CacheColumns.EXPERIMENT_NAME, experiment_name, timeout=timeout)
        cache.set(CacheColumns.EXPERIMENT_PATH, f'{OUTPUT_DIR}/{experiment_name}', timeout=timeout)
        return f'Experiment {experiment_name} started for {timeout/60} minutes'


@app.route('/get_experiment')
def get_experiment():
    return Response(cache.get(CacheColumns.EXPERIMENT_NAME))


@app.route('/stop_experiment')
def stop_experiment():
    experiment_name = cache.get(CacheColumns.EXPERIMENT_NAME)
    if experiment_name:
        cache.delete(CacheColumns.EXPERIMENT_NAME)
        return Response(f'Experiment: {experiment_name} was stopped')
    return Response('No available experiment')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    vc = VideoStream()
    return Response(gen(vc),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/manual_record_stop')
def manual_record_stop():
    cache.set(CacheColumns.MANUAL_RECORD_STOP, True)
    return Response('Record stopped')


@app.route('/init_bugs')
def init_bugs():
    mqtt_client.publish_event('event/command/init_bugs', 1)
    return Response('ok')


@app.route('/hide_bugs')
def hide_bugs():
    mqtt_client.publish_event('event/command/hide_bugs', '')
    return Response('ok')


class VideoStream:
    def __init__(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        filter_cameras(self.cam_list, 'left')
        if len(self.cam_list) == 0:
            self.clear()
            raise Exception('No cameras were found')

        self.sc = SpinCamera(self.cam_list[0])
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