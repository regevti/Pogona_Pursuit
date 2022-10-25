import yaml
from environs import Env
from pathlib import Path

env = Env()
env.read_env()

# General
version = '2.2'
is_debug_mode = env.bool('DEBUG', False)
is_use_parport = env.bool('IS_USE_PARPORT', False)

# API
static_files_dir = env('STATIC_FILES_DIR', 'static')
api_max_blocks_to_show = 20
LOGGING_LEVEL = env('LOGGING_LEVEL', 'DEBUG')
UI_LOGGING_LEVEL = env('UI_LOGGING_LEVEL', 'INFO')
FLASK_PORT = env.int('FLASK_PORT', 5000)
POGONA_HUNTER_PORT = env.int('POGONA_HUNTER_PORT', 8080)
management_url = env('MANAGEMENT_URL', f'http://localhost:{FLASK_PORT}')

# Cache (Redis)
redis_host = env('REDIS_HOST', 'cache')
websocket_url = env('WEBSOCKET_URL', 'ws://localhost:6380')
ui_console_channel = "cmd/visual_app/console"
# listeners that should listen only during an experiment
experiment_metrics = {
    'touch': {
        'is_write_csv': True,
        'is_write_db': True,
        'csv_file': 'screen_touches.csv',
        'is_overall_experiment': False
    },
    'trial_data': {
        'is_write_csv': True,
        'is_write_db': True,
        'csv_file': {'bug_trajectory': 'bug_trajectory.csv', 'video_frames': 'video_frames.csv'},
        'is_overall_experiment': False
    }
}
# listeners that should listen as long as the arena_mgr is alive
commands_topics = {
    'reward': 'cmd/arena/reward',
    'led_light': 'cmd/arena/led_light',
    'heat_light': 'cmd/arena/heat_light',

    'arena_shutdown': 'cmd/management/arena_shutdown',
    'end_experiment': 'cmd/management/end_experiment',

    'init_bugs': 'cmd/visual_app/init_bugs',
    'init_media': 'cmd/visual_app/init_media',
    'hide_bugs': 'cmd/visual_app/hide_bugs',
    'hide_media': 'cmd/visual_app/hide_media',
    'reload_app': 'cmd/visual_app/reload_app',
    'healthcheck': 'cmd/visual_app/healthcheck'
}
subscription_topics = {
    'arena_operations': 'cmd/arena/*',
    'metrics_logger': 'log/metric/*',
    'temperature': 'log/metric/temperature'
}
metric_channel_prefix = 'log/metric'
subscription_topics.update({k: f'{metric_channel_prefix}/{k}' for k in experiment_metrics.keys()})
subscription_topics.update(commands_topics)

# Arena
default_exposure = 5000
cameras = yaml.load(Path('cam_config.yaml').open(), Loader=yaml.FullLoader)

# Multi-Processing
array_queue_size_mb = env.int('ARRAY_QUEUE_SIZE_MB', 5 * 20)  # I assume that one image is roughly 5Mb
count_timestamps_for_fps_calc = env.int('count_timestamps_for_fps_calc', 200)  # how many timestamps to gather for calculating FPS
writing_video_queue_maxsize = env.int('writing_video_queue_maxsize', 100)
shm_buffer_dtype = 'uint8'

# Arena
arena_modules = {
    'cameras': {
        'allied_vision': ('cameras.allied_vision', 'AlliedVisionCamera'),
    },
    'predictors': {
        'pogona_head': ('image_handlers.pogona_head', 'PogonaHeadDetector'),
        'resnet': ('image_handlers.resnet_embedding', 'ResnetEmbedding'),
        'deeplabcut': ('image_handlers.deeplabcut', 'DeepLabCut'),
    }
}
output_dir = env('OUTPUT_DIR', '../output/recordings')
output_dir_key = 'output_dir'
pixels2cm = 0.01833304668870419
temperature_logging_delay_sec = env.int('temperature_logging_delay_sec', 5)
max_video_time_sec = env.int('max_video_time_sec', 60*10)
max_predictor_rows = env.int('max_predictor_rows', 60*60*5)

# temperature sensor
SERIAL_PORT_TEMP = env('SERIAL_PORT_TEMP', '/dev/ttyACM0')
SERIAL_BAUD = env.int('SERIAL_BAUD', 9600)

# Experiments
experiments_dir = env('EXPERIMENTS_DIR', "../output/experiments")
explore_experiment_dir = env('EXPLORE_EXPERIMENT_DIR', experiments_dir)
extra_time_recording = env.int('EXTRA_TIME_RECORDING', 30)
time_between_blocks = env.int('time_between_blocks', 300)
experiments_timeout = env.int('EXPERIMENTS_TIMEOUT', 60 * 60)
reward_timeout = env.int('reward_timeout', 10)
experiment_cache_path = env('experiment_cache_path', 'cached_experiments')
experiment_types = {
    'bugs': ['reward_type', 'bug_types', 'reward_bugs', 'bug_speed', 'movement_type', 'time_between_bugs',
             'is_anticlockwise' 'target_drift', 'background_color', 'exit_hole_position'],
    'media': ['media_url'],
    'blank': []
}
reward_types = [
    'always',
    'end_trial'
]

# Database
db_name = env('DB_NAME', 'arena')
db_host = env('DB_HOST', 'localhost')
db_port = env.int('DB_PORT', 5432)
db_engine = env('DB_ENGINE', 'postgresql+psycopg2')
db_user = env('DB_USER', 'postgres')
db_password = env('DB_PASSWORD', 'password')
sqlalchemy_url = f'{db_engine}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'