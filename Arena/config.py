from environs import Env
env = Env()
env.read_env()

# General
version = '1.1'
is_debug_mode = env.bool('DEBUG', False)
is_use_parport = env.bool('IS_USE_PARPORT', False)

# Experiments
experiments_dir = env('EXPERIMENTS_DIR', "experiments")
explore_experiment_dir = env('EXPLORE_EXPERIMENT_DIR', experiments_dir)
extra_time_recording = env.int('EXTRA_TIME_RECORDING', 30)
reward_types = [
    'always',
    'end_trial'
]

# Cache (Redis)
redis_host = env('REDIS_HOST', 'cache')

# MQTT
mqtt_host = env('MQTT_HOST', 'mqtt')
experiment_topic = "event/log/experiment"
log_topic_prefix = "event/log/"
logger_files = {
    'touch': 'screen_touches.csv',
    'prediction': 'predictions.csv',
    'trajectory': 'bug_trajectory.csv',
    'temperature': 'temperature.csv'
}
subscription_topics = {
    'reward': 'event/command/reward',
    'led_light': 'event/command/led_light',
    'end_trial': 'event/command/end_trial',
    'end_experiment': 'event/command/end_experiment',
    'touch': 'event/log/touch',
    'hit': 'event/log/hit',
    'prediction': 'event/log/prediction',
    'trajectory': 'event/log/trajectory',
    'temperature': 'event/log/temperature'
}

# Arena
default_num_frames = 1000
default_max_throughput = 94578303
exposure_time = env.int('EXPOSURE_TIME', 8000)
fps = env.int('FPS', 60)
output_dir = env('OUTPUT_DIR', 'output')
saved_frame_resolution = env.list('SAVED_FRAME_RESOLUTION', [1440, 1088])
camera_names = {
    'realtime': '19506468',
    'right': '19506475',
    'left': '19506455',
    'back': '19506481',
}
acquire_stop_options = {
    'num_frames': int,
    'record_time': int,
    'manual_stop': 'cache',
    'trial_alive': 'cache'
}
info_fields = [
    'AcquisitionFrameRate',
    'AcquisitionMode',
    'TriggerSource',
    'TriggerMode',
    'TriggerSelector',
    'PayloadSize',
    'EventSelector',
    'LineStatus',
    'ExposureTime',
    'DeviceLinkCurrentThroughput',
    'DeviceLinkThroughputLimit',
    'DeviceMaxThroughput',
    'DeviceLinkSpeed',
]

# Real-time Predictor
detector_thresh = env.float('DETECTOR_THRESH', 0.9)
realtime_camera = env('REALTIME_CAMERA', 'realtime')
is_disable_predictor = env.bool('DISABLE_PREDICTOR', False)
is_predictor_experiment = env.bool('PREDICTOR_EXPERIMENT', False)
predictor_model = env('PREDICTOR_MODEL', 'lstm')
