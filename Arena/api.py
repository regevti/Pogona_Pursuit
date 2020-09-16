from flask import Flask, render_template, Response, request, make_response, send_file, stream_with_context
from dotenv import load_dotenv
import PySpin
import cv2
import json
import os
load_dotenv()

from utils import titlize, get_predictor_model
from cache import RedisCache, CacheColumns
from mqtt import MQTTClient, SUBSCRIPTION_TOPICS
from experiment import Experiment, REWARD_TYPES
from arena import SpinCamera, record, capture_image, filter_cameras, display_info, \
    CAMERA_NAMES, EXPOSURE_TIME, ACQUIRE_STOP_OPTIONS

app = Flask(__name__)
cache = RedisCache()
mqtt_client = MQTTClient()


@app.route('/')
def index():
    """Video streaming ."""
    print(f'current dir: {os.getcwd()}')
    with open('../pogona_hunter/src/config.json', 'r') as f:
        config = json.load(f)
    return render_template('index.html', cameras=CAMERA_NAMES.keys(), exposure=EXPOSURE_TIME, config=config,
                           acquire_stop={k: titlize(k) for k in ACQUIRE_STOP_OPTIONS.keys()}, reward_types=REWARD_TYPES)


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
        data['cache'] = cache
        e = Experiment(**data)
        return Response(stream_with_context(e.start()))


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


@app.route('/calibrate')
def calibrate():
    """Calibrate camera"""
    try:
        from arena import _models
    except ImportError:
        return Response('Unable to locate HitPredictor')
    pred = _models[get_predictor_model()].hit_pred
    img = capture_image('realtime')
    h, h_im, error = pred.calibrate(img)
    if error:
        return Response(error)
    return Response('Calibration completed')


@app.route('/reward')
def reward():
    """Activate Feeder"""
    mqtt_client.publish_event(SUBSCRIPTION_TOPICS['reward'], '')
    return Response('ok')


@app.route('/cameras_info')
def cameras_info():
    """Get cameras info"""
    return Response(display_info())


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


@app.route('/set_stream_camera', methods=['POST'])
def set_stream_camera():
    if request.method == 'POST':
        cache.set(CacheColumns.STREAM_CAMERA, request.form['camera'])
        return Response(request.form['camera'])


@app.route('/init_bugs')
def init_bugs():
    mqtt_client.publish_event('event/command/init_bugs', 1)
    return Response('ok')


@app.route('/hide_bugs')
def hide_bugs():
    mqtt_client.publish_event('event/command/hide_bugs', '')
    return Response('ok')


class VideoStream:
    def __init__(self, exposure_time=EXPOSURE_TIME):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        filter_cameras(self.cam_list, cache.get(CacheColumns.STREAM_CAMERA))
        if len(self.cam_list) == 0:
            self.clear()
            raise Exception('No cameras were found')

        self.sc = SpinCamera(self.cam_list[0])
        self.sc.begin_acquisition(exposure_time)
        # self.serializer = Serializer()
        # self.serializer.start_acquisition()

    def get_frame(self):
        image_result = self.sc.cam.GetNextImage()
        img = image_result.GetNDArray()
        return cv2.imencode(".jpg", img)

    def clear(self):
        self.cam_list.Clear()
        del self.sc
        # self.system.ReleaseInstance()
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