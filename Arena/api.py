from flask import Flask, render_template, Response, request, send_from_directory
import PySpin
import cv2
import json
import os
import config
from pathlib import Path
from utils import titlize, turn_display_on, turn_display_off
from cache import RedisCache, CacheColumns
from mqtt import MQTTPublisher
from experiment import Experiment
from explore import ExperimentAnalyzer
from arena import SpinCamera, record, capture_image, filter_cameras, display_info

app = Flask(__name__)
cache = RedisCache()
mqtt_client = MQTTPublisher()


@app.route('/')
def index():
    """Video streaming ."""
    print(f'current dir: {os.getcwd()}')
    with open('../pogona_hunter/src/config.json', 'r') as f:
        app_config = json.load(f)
    return render_template('index.html', cameras=config.camera_names.keys(), exposure=config.exposure_time,
                           config=app_config, acquire_stop={k: titlize(k) for k in config.acquire_stop_options.keys()},
                           reward_types=config.reward_types, experiment_types=config.experiment_types,
                           media_files=list_media())


@app.route('/explore')
def explore():
    return render_template('explore.html')


@app.route('/experiment_results', methods=['POST'])
def experiment_results():
    data = request.json
    ea = ExperimentAnalyzer(**data)
    df = ea.get_experiments()
    if len(df) < 1:
        return Response('No experiments found')
    return Response(df.to_html(classes='table-responsive'))


@app.route('/record', methods=['POST'])
def record_video():
    """Record video"""
    if request.method == 'POST':
        data = request.json
        data.update({'cache': cache})
        return Response(record(**data))


@app.route('/start_experiment', methods=['POST'])
def start_experiment():
    """Set Experiment Name"""
    data = request.json
    e = Experiment(**data)
    return Response(e.start())


@app.route('/get_experiment')
def get_experiment():
    return Response(cache.get(CacheColumns.EXPERIMENT_NAME))


@app.route('/stop_experiment')
def stop_experiment():
    experiment_name = cache.get(CacheColumns.EXPERIMENT_NAME)
    mqtt_client.publish_command('end_experiment')
    if experiment_name:
        return Response(f'ending experiment {experiment_name}...')
    return Response('No available experiment')


@app.route('/calibrate')
def calibrate():
    """Calibrate camera"""
    try:
        from arena import _models
    except ImportError:
        return Response('Unable to locate HitPredictor')
    pred = _models[config.predictor_model].hit_pred
    img = capture_image('realtime')
    h, h_im, error = pred.calibrate(img)
    if error:
        return Response(error)
    return Response('Calibration completed')


@app.route('/reward')
def reward():
    """Activate Feeder"""
    mqtt_client.publish_event(config.subscription_topics['reward'], '')
    return Response('ok')


@app.route('/led_light/<state>')
def led_light(state):
    mqtt_client.publish_event(config.subscription_topics['led_light'], state)
    return Response('ok')


@app.route('/display/<state>')
def display(state):
    if state == 'off':
        stdout = turn_display_off()
    else:
        stdout = turn_display_on()
    return Response(stdout)


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


@app.route('/init_bugs', methods=['POST'])
def init_bugs():
    if request.method == 'POST':
        mqtt_client.publish_event('event/command/init_bugs', request.data.decode())
    return Response('ok')


@app.route('/hide_bugs')
def hide_bugs():
    mqtt_client.publish_event('event/command/hide_bugs', '')
    return Response('ok')


def list_media():
    media_files = []
    for f in Path(config.static_files_dir).glob('*'):
        if f.suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.avi', '.mp4', '.mpg']:
            media_files.append(f.name)
    return media_files


@app.route('/media/<filename>')
def send_media(filename):
    return send_from_directory(config.static_files_dir, filename)


class VideoStream:
    def __init__(self, exposure_time=config.exposure_time):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        filter_cameras(self.cam_list, cache.get(CacheColumns.STREAM_CAMERA))
        if len(self.cam_list) == 0:
            self.clear()
            raise Exception('No cameras were found')

        self.sc = SpinCamera(self.cam_list[0])
        self.sc.begin_acquisition(exposure_time)

    def get_frame(self):
        image_result = self.sc.cam.GetNextImage()
        img = image_result.GetNDArray()
        return cv2.imencode(".jpg", img)

    def clear(self):
        self.cam_list.Clear()
        del self.sc
        # self.system.ReleaseInstance()

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
