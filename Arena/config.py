from environs import Env
env = Env()
env.read_env()

# General
version = '2.0'
is_debug_mode = env.bool('DEBUG', False)
is_use_parport = env.bool('IS_USE_PARPORT', False)

# Experiments
experiments_dir = env('EXPERIMENTS_DIR', "experiments")
explore_experiment_dir = env('EXPLORE_EXPERIMENT_DIR', experiments_dir)
extra_time_recording = env.int('EXTRA_TIME_RECORDING', 30)
time_between_blocks = env.int('time_between_blocks', 300)
experiment_cache_path = env('experiment_cache_path', 'cached_experiments')
experiment_types = {
    'bugs': ['reward_type', 'bug_types', 'reward_bugs', 'bug_speed', 'movement_type', 'time_between_bugs',
             'is_anticlockwise' 'target_drift', 'background_color', 'exit_hole_position'],
    'media': ['media_url']
}
reward_types = [
    'always',
    'end_trial'
]

# API
static_files_dir = env('STATIC_FILES_DIR', 'static')
management_url = env('MANAGEMENT_URL', 'http://localhost:3351')
max_blocks = 20

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
    'temperature': 'temperature.csv',
    'video_frames': 'video_frames.csv'
}
subscription_topics = {
    'reward': 'event/command/reward',
    'led_light': 'event/command/led_light',
    'end_app_wait': 'event/command/end_app_wait',
    'end_experiment': 'event/command/end_experiment',
    'gaze_external': 'event/command/gaze_external',
    'touch': 'event/log/touch',
    'hit': 'event/log/hit',
    'prediction': 'event/log/prediction',
    'trajectory': 'event/log/trajectory',
    'temperature': 'event/log/temperature',
    'video_frames': 'event/log/video_frames'
}

# Arena
pixels2cm = 0.01833304668870419
default_num_frames = 1000
default_max_throughput = 94578303
exposure_time = env.int('EXPOSURE_TIME', 8000)
fps = env.int('FPS', 60)
output_dir = env('OUTPUT_DIR', 'output')
saved_frame_resolution = env.list('SAVED_FRAME_RESOLUTION', [1440, 1088])
camera_names = {
    'realtime': '19506468',
    'left': '19506455',
    'stream': '19506481',
    'back': '19506475',
    'top': '20349303'
}
acquire_stop_options = {
    'num_frames': int,
    'record_time': int,
    'manual_stop': 'cache',
    'trial_alive': 'cache',
    'thread_event': 'event'
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
