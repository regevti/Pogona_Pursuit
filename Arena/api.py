import json
import psutil
import logging
import torch.multiprocessing as mp
from pathlib import Path
from flask import Flask, render_template, Response, request, send_from_directory, jsonify
import config
from cache import RedisCache, CacheColumns as cc
from utils import titlize, turn_display_on, turn_display_off, get_sys_metrics
from experiment import ExperimentCache
from arena import ArenaManager
from loggers import init_logger_config
from db_models import ORM
import calibration

app = Flask('ArenaAPI')
cache = None
arena_mgr = None
orm = None


@app.route('/')
def index():
    """Video streaming ."""
    cached_experiments = [c.stem for c in Path(config.experiment_cache_path).glob('*.json')]
    with open('../pogona_hunter/src/config.json', 'r') as f:
        app_config = json.load(f)
    if arena_mgr is None:
        cameras = list(config.cameras.keys())
    else:
        cameras = list(arena_mgr.detected_cameras.keys())
    return render_template('index.html', cameras=cameras, exposure=config.default_exposure,
                           config=app_config, log_channel=config.ui_console_channel, reward_types=config.reward_types,
                           experiment_types=config.experiment_types, media_files=list_media(),
                           max_blocks=config.api_max_blocks_to_show, cached_experiments=cached_experiments,
                           extra_time_recording=config.extra_time_recording,
                           acquire_stop={'num_frames': 'Num Frames', 'rec_time': 'Record Time [sec]'})


@app.route('/check', methods=['GET'])
def check():
    res = dict()
    res['experiment_name'] = cache.get_current_experiment()
    res['block_id'] = cache.get(cc.EXPERIMENT_BLOCK_ID)
    res['open_app_host'] = cache.get(cc.OPEN_APP_HOST)
    res['temperature'] = orm.get_temperature()
    for cam_name, cu in arena_mgr.units.items():
        res.setdefault('cam_units_status', {})[cam_name] = cu.is_on()
        res.setdefault('cam_units_fps', {})[cam_name] = {k: cu.mp_metadata.get(k).value for k in ['cam_fps', 'sink_fps', 'pred_fps', 'pred_delay']}
        res.setdefault('cam_units_predictors', {})[cam_name] = ','.join(cu.get_alive_predictors()) or '-'
        proc_cpus = {}
        for p in cu.processes.copy().values():
            try:
                proc_cpus[p.name] = psutil.Process(p.pid).cpu_percent(0.1)
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
    return Response('ok')


@app.route('/calibrate')
def calibrate():
    """Calibrate camera"""
    img = arena_mgr.get_frame('front')


@app.route('/calibrate_aruco')
def calibrate_aruco():
    """Calibrate camera"""
    img = arena_mgr.get_frame('front')
    try:
        h, _, h_im, error = calibration.calibrate(img)
    except Exception as exc:
        arena_mgr.logger.error(f'Error in calibrate; {exc}')
        error = str(exc)

    if error:
        return Response(error)
    arena_mgr.logger.info('calibration completed')
    return Response('Calibration completed')


@app.route('/reward')
def reward():
    """Activate Feeder"""
    cache.publish_command('reward')
    return Response('ok')


@app.route('/led_light/<state>')
def led_light(state):
    cache.publish_command('led_light', state)
    return Response('ok')


@app.route('/heat_light/<state>')
def heat_light(state):
    cache.publish_command('heat_light', state)
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


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(arena_mgr.stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def initialize():
    logger = logging.getLogger(app.name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("""%(levelname)s in %(module)s [%(pathname)s:%(lineno)d]:\n%(message)s""")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


if __name__ == "__main__":
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
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)

    import torch
    torch.cuda.set_device(0)

    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    init_logger_config()
    cache = RedisCache()
    arena_mgr = ArenaManager()
    orm = ORM()

    app.run(host='0.0.0.0', port=config.FLASK_PORT, debug=False)
