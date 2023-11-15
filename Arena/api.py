import io
import time

import cv2
import json
import warnings
import base64
import psutil
import logging
import pytest
from pathlib import Path
from PIL import Image
from datetime import datetime
import torch.multiprocessing as mp
from flask import Flask, render_template, Response, request, send_from_directory, jsonify, send_file
import sentry_sdk
import config
import utils
from cache import RedisCache, CacheColumns as cc
from utils import titlize, turn_display_on, turn_display_off, get_sys_metrics, get_psycho_files
from experiment import ExperimentCache
from arena import ArenaManager
from loggers import init_logger_config, create_arena_handler
from calibration import CharucoEstimator
from periphery_integration import PeripheryIntegrator
from agent import Agent
import matplotlib
matplotlib.use('Agg')

app = Flask('ArenaAPI')
cache = None
arena_mgr = None
periphery_mgr = None
queue_app = None


@app.route('/')
def index():
    """Video streaming ."""
    cached_experiments = sorted([c.stem for c in Path(config.experiment_cache_path).glob('*.json')])
    with open('../pogona_hunter/src/config.json', 'r') as f:
        app_config = json.load(f)
    if arena_mgr is None:
        cameras = list(config.cameras.keys())
    else:
        cameras = list(arena_mgr.units.keys())
    if config.IS_ANALYSIS_ONLY:
        toggels, feeders = [], []
    else:
        toggels, feeders = periphery_mgr.toggles, periphery_mgr.feeders
    return render_template('index.html', cameras=cameras, exposure=config.DEFAULT_EXPOSURE, arena_name=config.ARENA_NAME,
                           config=app_config, log_channel=config.ui_console_channel, reward_types=config.reward_types,
                           experiment_types=config.experiment_types, media_files=list_media(),
                           blank_rec_types=config.blank_rec_types,
                           max_blocks=config.api_max_blocks_to_show, toggels=toggels, psycho_files=get_psycho_files(),
                           extra_time_recording=config.extra_time_recording, feeders=feeders,
                           acquire_stop={'num_frames': 'Num Frames', 'rec_time': 'Record Time [sec]'})


@app.route('/check', methods=['GET'])
def check():
    # periphery_mgr.publish_cam_trigger_state()
    res = dict()
    res['experiment_name'] = cache.get_current_experiment()
    res['block_id'] = cache.get(cc.EXPERIMENT_BLOCK_ID)
    res['open_app_host'] = cache.get(cc.OPEN_APP_HOST)
    if not config.DISABLE_DB:
        res['temperature'] = arena_mgr.orm.get_temperature()
        res['n_strikes'] = sum(arena_mgr.orm.get_today_strikes().values())
        rewards_dict = arena_mgr.orm.get_today_rewards()
        res['n_rewards'] = f'{rewards_dict["auto"]} ({rewards_dict["manual"]})'
    else:
        res.update({'temperature': None, 'n_strikes': 0, 'n_rewards': 0})
    res['reward_left'] = periphery_mgr.get_feeders_counts()
    res['streaming_camera'] = arena_mgr.get_streaming_camera()
    res['schedules'] = arena_mgr.schedules
    res['cached_experiments'] = sorted([c.stem for c in Path(config.experiment_cache_path).glob('*.json')])
    res['cam_trigger_state'] = cache.get(cc.CAM_TRIGGER_STATE)
    for cam_name, cu in arena_mgr.units.copy().items():
        res.setdefault('cam_units_status', {})[cam_name] = cu.is_on()
        res.setdefault('cam_units_fps', {})[cam_name] = {k: cu.mp_metadata.get(k).value for k in ['cam_fps', 'sink_fps', 'pred_fps', 'pred_delay']}
        res.setdefault('cam_units_predictors', {})[cam_name] = ','.join(cu.get_alive_predictors()) or '-'
        proc_cpus = {}
        for p in cu.processes.copy().values():
            try:
                proc_cpus[p.name] = round(psutil.Process(p.pid).cpu_percent(0.1))
            except:
                continue
        res.setdefault('processes_cpu', {})[cam_name] = proc_cpus
    res.update(get_sys_metrics())
    return jsonify(res)


@app.route('/record', methods=['POST'])
def record_video():
    """Record video"""
    if request.method == 'POST':
        data = request.json
        return Response(arena_mgr.record(**data))


@app.route('/start_experiment', methods=['POST'])
def start_experiment():
    """Set Experiment Name"""
    data = request.json
    print(data)
    e = arena_mgr.start_experiment(**data)
    return Response(e)


@app.route('/save_experiment', methods=['POST'])
def save_experiment():
    """Set Experiment Name"""
    data = request.json
    ExperimentCache().save(data)
    return Response('ok')


@app.route('/load_experiment/<name>')
def load_experiment(name):
    """Load Cached Experiment"""
    data = ExperimentCache().load(name)
    return jsonify(data)


@app.route('/get_experiment')
def get_experiment():
    return Response(cache.get_current_experiment())


@app.route('/stop_experiment')
def stop_experiment():
    experiment_name = cache.get_current_experiment()
    if experiment_name:
        cache.stop_experiment()
        return Response(f'ending experiment {experiment_name}...')
    return Response('No available experiment')


@app.route('/commit_schedule', methods=['POST'])
def commit_schedule():
    data = dict(request.form)
    if not data.get('start_date'):
        arena_mgr.logger.error('please enter start_date for schedule')
    else:
        data['start_date'] = datetime.strptime(data['start_date'], '%d/%m/%Y %H:%M')
        if data.get('end_date'):
            data['end_date'] = datetime.strptime(data['end_date'], '%d/%m/%Y %H:%M')
        data['every'] = int(data['every'])
        arena_mgr.orm.commit_multiple_schedules(**data)
        arena_mgr.update_upcoming_schedules()
    return Response('ok')


@app.route('/delete_schedule', methods=['POST'])
def delete_schedule():
    arena_mgr.orm.delete_schedule(request.form['schedule_id'])
    arena_mgr.update_upcoming_schedules()
    return Response('ok')


@app.route('/update_reward_count', methods=['POST'])
def update_reward_count():
    data = request.json
    feeder_name = data.get('name')
    reward_count = int(data.get('reward_count', 0))
    arena_mgr.logger.info(f'Update {feeder_name} to {reward_count}')
    periphery_mgr.update_reward_count(feeder_name, reward_count)
    return Response('ok')


@app.route('/update_animal_id', methods=['POST'])
def update_animal_id():
    data = request.json
    animal_id = data['animal_id']
    current_animal_id = cache.get(cc.CURRENT_ANIMAL_ID)
    if animal_id != current_animal_id:
        if current_animal_id:
            arena_mgr.orm.update_animal_id(end_time=datetime.now())
        if animal_id:
            arena_mgr.orm.commit_animal_id(**data)
            arena_mgr.logger.info(f'Animal ID was updated to {animal_id} ({data["sex"]})')
    else:
        arena_mgr.orm.update_animal_id(**data)
    return Response('ok')


@app.route('/get_current_animal', methods=['GET'])
def get_current_animal():
    if config.DISABLE_DB:
        return jsonify({})
    animal_id = cache.get(cc.CURRENT_ANIMAL_ID)
    if not animal_id:
        arena_mgr.logger.warning('No animal ID is set')
        return jsonify({})
    animal_dict = arena_mgr.orm.get_animal_settings(animal_id)
    return jsonify(animal_dict)


@app.route('/start_camera_unit', methods=['POST'])
def start_camera_unit():
    cam_name = request.form['camera']
    if cam_name not in arena_mgr.units:
        app.logger.error(f'cannot start camera unit {cam_name} - unknown')
        return Response('')

    arena_mgr.units[cam_name].start()
    return Response('ok')


@app.route('/stop_camera_unit', methods=['POST'])
def stop_camera_unit():
    cam_name = request.form['camera']
    if cam_name not in arena_mgr.units:
        app.logger.error(f'cannot start camera unit {cam_name} - unknown')
        return Response('')

    arena_mgr.units[cam_name].stop()
    if cam_name == arena_mgr.get_streaming_camera():
        arena_mgr.stop_stream()
    return Response('ok')


@app.route('/set_cam_trigger', methods=['POST'])
def set_cam_trigger():
    if cache.get(cc.CAM_TRIGGER_DISABLE):
        # during experiments the trigger gui is disabled
        return Response('ok')
    state = int(request.form['state'])
    periphery_mgr.cam_trigger(state)
    return Response('ok')


@app.route('/update_trigger_fps', methods=['POST'])
def update_trigger_fps():
    data = request.json
    periphery_mgr.change_trigger_fps(data['fps'])
    return Response('ok')


@app.route('/capture', methods=['POST'])
def capture():
    cam = request.form['camera']
    folder_prefix = request.form.get('folder_prefix')
    img = arena_mgr.get_frame(cam)
    dir_path = config.capture_images_dir
    Path(dir_path).mkdir(exist_ok=True, parents=True)
    if folder_prefix:
        dir_path = Path(dir_path) / folder_prefix
        dir_path.mkdir(exist_ok=True, parents=True)
    img_path = f'{dir_path}/{utils.datetime_string()}_{cam}.png'
    cv2.imwrite(img_path, img)
    arena_mgr.logger.info(f'Image from {cam} was saved to: {img_path}')
    return Response('ok')


@app.route('/calibrate', methods=['POST'])
def calibrate():
    """Calibrate camera"""
    cam = request.form['camera']
    img = arena_mgr.get_frame(cam)
    conf_preds = arena_mgr.units[cam].get_conf_predictors()
    pred_image_size = list(conf_preds.values())[0][:2] if conf_preds else img.shape[:2]
    try:
        pe = CharucoEstimator(cam, pred_image_size)
        img, ret = pe.find_aruco_markers(img)
    except Exception as exc:
        arena_mgr.logger.error(f'Error in calibrate; {exc}')
        return Response('error')
    img = img.astype('uint8')

    # cv2.imwrite(f'../output/calibrations/{cam}.jpg', img)
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = Image.fromarray(img)
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'status': str(img_base64)})


@app.route('/reward')
def reward():
    """Activate Feeder"""
    # cache.publish_command('reward')
    periphery_mgr.feed(is_manual=True)
    return Response('ok')


@app.route('/arena_switch/<name>/<state>')
def arena_switch(name, state):
    state = int(state)
    assert state in [0, 1], f'state must be 0 or 1; received {state}'
    arena_mgr.logger.debug(f'Turn {name} {"on" if state == 1 else "off"}')
    periphery_mgr.switch(name, state)
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
    return Response(arena_mgr.display_info(return_string=True))


@app.route('/check_cameras')
def check_cameras():
    """Check all cameras are connected"""
    if config.is_debug_mode:
        return Response(json.dumps([]))
    info_df = arena_mgr.display_info()
    missing_cameras = []
    for cam_name, cam_config in config.cameras.items():
        if cam_name not in info_df.index:
            missing_cameras.append(cam_name)

    return Response(json.dumps(missing_cameras))


@app.route('/get_camera_settings/<name>')
def get_camera_settings(name):
    return jsonify(arena_mgr.units[name].cam_config)


@app.route('/update_camera/<name>', methods=['POST'])
def update_camera_settings(name):
    data = request.json
    arena_mgr.update_camera_unit(name, data)
    return Response('ok')


@app.route('/manual_record_stop')
def manual_record_stop():
    arena_mgr.stop_recording()
    return Response('Record stopped')


@app.route('/reload_app')
def reload_app():
    cache.publish_command('reload_app')


@app.route('/init_bugs', methods=['POST'])
def init_bugs():
    if request.method == 'POST':
        cache.publish_command('init_bugs', request.data.decode())
    return Response('ok')


@app.route('/hide_bugs')
def hide_bugs():
    cache.publish_command('hide_bugs', '')
    return Response('ok')


@app.route('/start_media', methods=['POST'])
def start_media():
    if request.method == 'POST':
        data = request.json
        if not data or not data.get('media_url'):
            return Response('Unable to find media url')
        payload = json.dumps({'url': f'{config.management_url}/media/{data["media_url"]}'})
        print(payload)
        cache.publish_command('init_media', payload)
    return Response('ok')


@app.route('/stop_media')
def stop_media():
    cache.publish_command('hide_media')
    return Response('ok')


def list_media():
    media_files = []
    for f in Path(config.static_files_dir).glob('*'):
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.avi', '.mp4', '.mpg', '.mov']:
            media_files.append(f.name)
    return media_files


@app.route('/media/<filename>')
def send_media(filename):
    return send_from_directory(config.static_files_dir, filename)


@app.route('/set_stream_camera', methods=['POST'])
def set_stream_camera():
    if request.method == 'POST':
        arena_mgr.set_streaming_camera(request.form['camera'])
        return Response(request.form['camera'])


@app.route('/stop_stream_camera', methods=['POST'])
def stop_stream_camera():
    if request.method == 'POST':
        arena_mgr.stop_stream()
        return Response('ok')


@app.route('/animal_summary', methods=['GET'])
def animal_summary():
    ag = Agent()
    ag.update()
    return Response(ag.get_animal_history())


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(arena_mgr.stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/strike_analysis/<strike_id>')
def get_strike_analysis(strike_id):
    from analysis.strikes import StrikeAnalyzer, Loader
    ld = Loader(strike_id, 'front', is_debug=False)
    sa = StrikeAnalyzer(ld)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = sa.plot_strike_analysis(only_return=True)
    img = Image.fromarray(img.astype('uint8'))
    # create file-object in memory
    file_object = io.BytesIO()
    # write PNG in file-object
    img.save(file_object, 'PNG')
    # move to beginning of file so `send_file()` it will read from start
    file_object.seek(0)
    return send_file(file_object, mimetype='image/PNG')


def initialize():
    logger = logging.getLogger(app.name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("""%(levelname)s in %(module)s [%(pathname)s:%(lineno)d]:\n%(message)s""")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


import os
import re


def get_chunk(filename, byte1=None, byte2=None):
    filesize = os.path.getsize(filename)
    yielded = 0
    yield_size = 1024 * 1024

    if byte1 is not None:
        if not byte2:
            byte2 = filesize
        yielded = byte1
        filesize = byte2

    with open(filename, 'rb') as f:
        content = f.read()

    while True:
        remaining = filesize - yielded
        if yielded == filesize:
            break
        if remaining >= yield_size:
            yield content[yielded:yielded+yield_size]
            yielded += yield_size
        else:
            yield content[yielded:yielded+remaining]
            yielded += remaining


@app.route('/play_video11')
def get_file():
    filename = '/data/Pogona_Pursuit/Arena/static/back_20221106T093511.mp4'
    filesize = os.path.getsize(filename)
    range_header = request.headers.get('Range', None)

    if range_header:
        byte1, byte2 = None, None
        match = re.search(r'(\d+)-(\d*)', range_header)
        groups = match.groups()

        if groups[0]:
            byte1 = int(groups[0])
        if groups[1]:
            byte2 = int(groups[1])

        if not byte2:
            byte2 = byte1 + 1024 * 1024
            if byte2 > filesize:
                byte2 = filesize

        length = byte2 + 1 - byte1

        resp = Response(
            get_chunk(filename, byte1, byte2),
            status=206, mimetype='video/mp4',
            content_type='video/mp4',
            direct_passthrough=True
        )

        resp.headers.add('Content-Range',
                         'bytes {0}-{1}/{2}'
                         .format(byte1,
                                 length,
                                 filesize))
        return resp

    return Response(
        get_chunk(filename),
        status=200, mimetype='video/mp4'
    )


@app.after_request
def after_request(response):
    response.headers.add('Accept-Ranges', 'bytes')
    return response


@app.route('/play_video')
def play():
    return render_template('management/play_video.html')


@app.route('/restart')
def restart():
    arena_mgr.arena_shutdown()
    queue_app.put('restart')
    return Response('ok')


def start_app(queue):
    global cache, arena_mgr, periphery_mgr, queue_app
    queue_app = queue

    import torch
    torch.cuda.set_device(0)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    init_logger_config()
    arena_handler = create_arena_handler('API')
    app.logger.addHandler(arena_handler)
    app.logger.setLevel(logging.INFO)

    cache = RedisCache()
    if not config.IS_ANALYSIS_ONLY:
        arena_mgr = ArenaManager()
        periphery_mgr = PeripheryIntegrator()
        utils.turn_display_off()
        if arena_mgr.is_cam_trigger_setup() and not config.DISABLE_PERIPHERY:
            periphery_mgr.cam_trigger(1)

    app.run(host='0.0.0.0', port=config.FLASK_PORT, debug=False)


if __name__ == "__main__":
    assert pytest.main(['-x', 'tests']) == 0

    # app.logger.removeHandler(flask_logging.default_handler)
    # h = logging.StreamHandler(sys.stdout)
    # h.setLevel(logging.WARNING)
    # h.setFormatter(CustomFormatter())
    # werklogger = logging.getLogger('werkzeug')
    # werklogger.addHandler(h)
    # app.debug = False
    # logger = logging.getLogger(app.name)
    # h = logging.StreamHandler()
    # h.setFormatter(CustomFormatter())
    # logger.addHandler(h)
    if not config.IS_ANALYSIS_ONLY and config.SENTRY_DSN:
        sentry_sdk.init(
            dsn=config.SENTRY_DSN,
            # Set traces_sample_rate to 1.0 to capture 100%
            # of transactions for performance monitoring.
            # We recommend adjusting this value in production.
            traces_sample_rate=1.0
        )

    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    queue_app = mp.Queue()

    while True:
        p = mp.Process(target=start_app, args=(queue_app,), name='MAIN')
        p.start()
        while True:
            if queue_app.empty():
                time.sleep(1)
            else:
                x = queue_app.get()
                break
        app.logger.warning('Restarting Arena!')
        p.terminate()

    app.run(host='0.0.0.0', port=config.FLASK_PORT, debug=False)

