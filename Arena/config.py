from environs import Env
from enum import Enum, auto

env = Env()
env.read_env()

# General
version = '2.2'
is_debug_mode = env.bool('DEBUG', False)
is_use_parport = env.bool('IS_USE_PARPORT', False)

# Experiments
experiments_dir = env('EXPERIMENTS_DIR', "experiments")
explore_experiment_dir = env('EXPLORE_EXPERIMENT_DIR', experiments_dir)
extra_time_recording = env.int('EXTRA_TIME_RECORDING', 30)
time_between_blocks = env.int('time_between_blocks', 300)
experiments_timeout = env.int('EXPERIMENTS_TIMEOUT', 60 * 60)
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
api_max_blocks_to_show = 20
LOGGING_LEVEL = env('LOGGING_LEVEL', 'DEBUG')
FLASK_PORT = env.int('FLASK_PORT', 5000)

# Cache (Redis)
redis_host = env('REDIS_HOST', 'cache')
ui_console_channel = "cmd/visual_app/console"
experiment_metrics = {
    'touch': {
        'is_write_csv': True,
        'is_write_db': True,
        'csv_file': 'screen_touches.csv',
        'is_overall_experiment': False
    },
    'trajectory': {
        'is_write_csv': True,
        'is_write_db': False,
        'csv_file': 'bug_trajectory.csv',
        'is_overall_experiment': False
    },
    'temperature': {
        'is_write_csv': False,
        'is_write_db': True,
        'is_overall_experiment': True
    },
    'video_frames': {
        'is_write_csv': True,
        'is_write_db': False,
        'csv_file': 'video_frames.csv',
        'is_overall_experiment': False
    },
    'trials_times': {
        'is_write_csv': False,
        'is_write_db': True,
        'is_overall_experiment': False
    },
}
commands_topics = {
    'reward': 'cmd/arena/reward',
    'led_light': 'cmd/arena/led_light',

    'start_recording': 'cmd/management/start_recording',
    'stop_recording': 'cmd/management/stop_recording',
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
}
metric_channel_prefix = 'log/metric'
subscription_topics.update({k: f'{metric_channel_prefix}/{k}' for k in experiment_metrics.keys()})
subscription_topics.update(commands_topics)

# Arena
default_exposure = 10000
cameras = {
    'color': {
        'id': 'DEV_1AB22C017E6D',
        'module': 'allied_vision',
        'fps': 80,
        'exposure': 12000,
        'image_size': [1088, 1456, 3],
        'listeners': ['video_writer']
    },
    'left': {
        'id': 'DEV_1AB22C017E70',
        'module': 'allied_vision',
        'fps': 80,
        'exposure': default_exposure,
        'image_size': [1088, 1456, 3],
        'listeners': ['video_writer']
    }
}
arena_modules = {
    'cameras': {
        'allied_vision': ('cameras.allied_vision', 'AlliedVisionCamera'),
    },
    'video_writer': ('image_handlers.video_writer', 'VideoWriter')
}
arena_manager_address = env('ARENA_MANAGER_ADDRESS', '127.0.0.1')
arena_manager_port = env.int('ARENA_MANAGER_PORT', 50000)
arena_manager_password = env('ARENA_MANAGER_PASSWORD', '123456')
output_dir = env('OUTPUT_DIR', 'output')
shm_buffer_dtype = 'uint8'
pixels2cm = 0.01833304668870419
default_num_frames = 1000
default_max_throughput = 94578303

# Database
db_name = env('DB_NAME', 'arena')
db_host = env('DB_HOST', 'localhost')
db_port = env.int('DB_PORT', 5432)
db_engine = env('DB_ENGINE', 'postgresql+psycopg2')
db_user = env('DB_USER', 'postgres')
db_password = env('DB_PASSWORD', 'password')
sqlalchemy_url = f'{db_engine}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

# Real-time Predictor
detector_thresh = env.float('DETECTOR_THRESH', 0.9)
realtime_camera = env('REALTIME_CAMERA', 'realtime')
is_disable_predictor = env.bool('DISABLE_PREDICTOR', False)
is_predictor_experiment = env.bool('PREDICTOR_EXPERIMENT', False)
predictor_model = env('PREDICTOR_MODEL', 'lstm')
