import yaml
from environs import Env
from pathlib import Path

env = Env()
env.read_env('configurations/.env')

# General
version = '2.2'
ARENA_NAME = env('ARENA_NAME')
is_debug_mode = env.bool('DEBUG', False)
is_use_parport = env.bool('IS_USE_PARPORT', False)
IS_ANALYSIS_ONLY = env.bool('IS_ANALYSIS_ONLY', False)

# API
static_files_dir = env('STATIC_FILES_DIR', 'static')
api_max_blocks_to_show = 20
LOGGING_LEVEL = env('LOGGING_LEVEL', 'DEBUG')
UI_LOGGING_LEVEL = env('UI_LOGGING_LEVEL', 'INFO')
FLASK_PORT = env.int('FLASK_PORT', 5084)
POGONA_HUNTER_PORT = env.int('POGONA_HUNTER_PORT', 8080)
SENTRY_DSN = env('SENTRY_DSN', '')
management_url = env('MANAGEMENT_URL', f'http://localhost:{FLASK_PORT}')
DISABLE_ARENA_SCREEN = env.bool('DISABLE_ARENA_SCREEN', 0)

# Cache (Redis)
redis_host = env('REDIS_HOST', 'localhost')
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
    'app_healthcheck': 'cmd/visual_app/healthcheck',
    'strike_predicted': 'cmd/visual_app/strike_predicted'
}
subscription_topics = {
    'arena_operations': 'cmd/arena/*',
    'metrics_logger': 'log/metric/*',
    'temperature': 'log/metric/temperature'
}
metric_channel_prefix = 'log/metric'
subscription_topics.update({k: f'{metric_channel_prefix}/{k}' for k in experiment_metrics.keys()})
subscription_topics.update(commands_topics)

# Multi-Processing
array_queue_size_mb = env.int('ARRAY_QUEUE_SIZE_MB', 5 * 20)  # I assume that one image is roughly 5Mb
count_timestamps_for_fps_calc = env.int('count_timestamps_for_fps_calc', 200)  # how many timestamps to gather for calculating FPS
writing_video_queue_maxsize = env.int('writing_video_queue_maxsize', 100)
shm_buffer_dtype = 'uint8'

# Arena
arena_modules = {
    'cameras': {
        'allied_vision': ('cameras.allied_vision', 'AlliedVisionCamera'),
        'flir': ('cameras.flir', 'FLIRCamera')
    },
    'image_handlers': {
        'pogona_head': ('image_handlers.predictor_handlers', 'PogonaHeadHandler'),
        'tongue_out': ('image_handlers.predictor_handlers', 'TongueOutHandler')
    },
    'predictors': {
        'deeplabcut': ('analysis.predictors.deeplabcut', 'DLCPose'),
        'tongue_out': ('analysis.predictors.tongue_out', 'TongueOutAnalyzer'),
        'pogona_head': ('analysis.predictors.pogona_head', 'PogonaHead')
    }
}
cameras = yaml.load(Path('configurations/cam_config.yaml').open(), Loader=yaml.FullLoader)
QUEUE_WAIT_TIME = env.int('QUEUE_WAIT_TIME', 2)
SINK_QUEUE_TIMEOUT = env.int('SINK_QUEUE_TIMEOUT', 2)
VIDEO_WRITER_FORMAT = env('VIDEO_WRITER_FORMAT', 'MJPG')
DEFAULT_EXPOSURE = env.int('DEFAULT_EXPOSURE', 5000)
camera_on_min_duration = env('camera_on_min_duration', 10*60)
ARENA_DISPLAY = env('ARENA_DISPLAY', ':0')
output_dir_key = 'output_dir'  # used for cam_config
pixels2cm = 0.01833304668870419
temperature_logging_delay_sec = env.int('temperature_logging_delay_sec', 5)
max_video_time_sec = env.int('max_video_time_sec', 60*10)
OUTPUT_DIR = env('OUTPUT_DIR', '/data/Pogona_Pursuit/output')
recordings_output_dir = env('recordings_output_dir', f'{OUTPUT_DIR}/recordings')
capture_images_dir = env('capture_images_dir', f'{OUTPUT_DIR}/captures')

# Periphery
DISABLE_PERIPHERY = env.bool('DISABLE_PERIPHERY', False)
mqtt = {
    'host': 'localhost',
    'port': env.int('MQTT_PORT', 1883),
    'publish_topic': 'arena_command',
    'temperature_sensors': env.list('TEMPERATURE_SENSORS', ['Temp'])
}
IR_NAME = env('IR_NAME', 'IR_lights')
LED_NAME = env('LED_NAME', 'day_lights')
FEEDER_NAME = env('FEEDER_NAME', 'Feeder 1')
TOUCH_SCREEN_NAME = env('TOUCH_SCREEN_NAME', 'Elo')
IS_SCREEN_INVERTED = env.bool('IS_SCREEN_INVERTED', 0)
IS_CHECK_SCREEN_MAPPING = env.bool('IS_CHECK_SCREEN_MAPPING', 1)
APP_SCREEN = env('APP_SCREEN', ':0.0')
TEST_SCREEN = env('TEST_SCREEN', ':1.0')
SCREEN_RESOLUTION = env('SCREEN_RESOLUTION', '1920,1080')  # must be written with comma
SCREEN_DISPLACEMENT = env('SCREEN_DISPLACEMENT', '2025')  # used for displacing the screen contents in multi screen setup
HOLD_TRIGGERS_TIME = env.int('HOLD_TRIGGERS_TIME', 2)

# temperature sensor
SERIAL_PORT_TEMP = env('SERIAL_PORT_TEMP', '/dev/ttyACM0')
SERIAL_BAUD = env.int('SERIAL_BAUD', 9600)

# Calibration
calibration_dir = env('calibration_dir', f'{OUTPUT_DIR}/calibrations')
min_calib_images = env.int('min_calib_images', 7)
CHESSBOARD_DIM = env.list('CHESSBOARD_DIM', (9, 6))
ARUCO_MARKER_SIZE = env.float('ARUCO_MARKER_SIZE', 2.65)  # centimeters

# Schedules
DISABLE_SCHEDULER = env.bool('DISABLE_SCHEDULER', False)
schedule_date_format = env('schedule_date_format', "%d/%m/%Y %H:%M")
IS_RUN_NIGHTLY_POSE_ESTIMATION = env.bool('IS_RUN_NIGHTLY_POSE_ESTIMATION', True)
MAX_COMPRESSION_THREADS = env.int('MAX_COMPRESSION_THREADS', 2)
IR_LIGHT_NAME = env('IR_LIGHT_NAME', '')
DAY_LIGHT_NAME = env('DAY_LIGHT_NAME', '')
SCHEDULE_EXPERIMENTS_END_TIME = env('SCHEDULE_EXPERIMENTS_END_TIME', '19:00')
IS_COMMIT_TO_DWH = env.bool('IS_COMMIT_TO_DWH', True)

# Experiments
EXPERIMENTS_DIR = env('EXPERIMENTS_DIR', f"{OUTPUT_DIR}/experiments")
explore_experiment_dir = env('EXPLORE_EXPERIMENT_DIR', EXPERIMENTS_DIR)
IS_RECORD_SCREEN_IN_EXPERIMENT = env.bool('IS_RECORD_SCREEN_IN_EXPERIMENT', True)
extra_time_recording = env.int('EXTRA_TIME_RECORDING', 30)
time_between_blocks = env.int('time_between_blocks', 300)
experiments_timeout = env.int('EXPERIMENTS_TIMEOUT', 60 * 60)
reward_timeout = env.int('reward_timeout', 10)
experiment_cache_path = env('experiment_cache_path', 'cached_experiments')
MAX_DURATION_CONT_BLANK = env.int('MAX_DURATION_CONT_BLANK', 48*3600)
experiment_types = {
    'bugs': ['reward_type', 'bug_types', 'reward_bugs', 'bug_speed', 'movement_type', 'time_between_bugs',
             'is_anticlockwise' 'target_drift', 'background_color', 'exit_hole_position'],
    'media': ['media_url'],
    'blank': ['blank_rec_type'],
    'psycho': ['psycho_file']
}
blank_rec_types = [
    'trials',
    'continuous'
]
reward_types = [
    'always',
    'end_trial'
]

# Database
DISABLE_DB = env.bool('DISABLE_DB', False)
db_name = env('DB_NAME', 'arena')
db_host = env('DB_HOST', 'localhost')
db_port = env.int('DB_PORT', 5432)
db_engine = env('DB_ENGINE', 'postgresql+psycopg2')
db_user = env('DB_USER', 'postgres')
db_password = env('DB_PASSWORD', 'password')
sqlalchemy_url = f'{db_engine}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
DWH_HOST = env('DWH_HOST', '132.66.212.143')
DWH_URL = f'{db_engine}://{db_user}:{db_password}@{DWH_HOST}:{db_port}/{db_name}'

# Telegram
TELEGRAM_CHAT_ID = env('TELEGRAM_CHAT_ID', '725002866')
TELEGRAM_TOKEN = env('TELEGRAM_TOKEN', None)

# Predictors
DLC_FOLDER = env('DLC_FOLDER', f'{OUTPUT_DIR}/models/deeplabcut')

# PsychoPy
PSYCHO_FOLDER = env('PSYCHO_FOLDER', '/data/Pogona_Pursuit/psycho_files')
PSYCHO_PYTHON_INTERPRETER = env('PSYCHO_PYTHON_INTERPRETER', '/home/regev/anaconda3/envs/psycho/bin/python')


class UserEnv:
    pass