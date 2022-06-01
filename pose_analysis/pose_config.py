from environs import Env

env = Env()
env.read_env()

MAIN_PATH = env('MAIN_PATH', '/data')

EXPERIMENTS_DIR = env('EXPERIMENTS_DIR', MAIN_PATH + '/Pogona_Pursuit/Arena/experiments')
CAMERAS = {
    'realtime': '19506468',
    'right': '19506475',
    'left': '19506455',
    'back': '19506481',
}
SCREEN_BOUNDARIES = {'x': (0, 1850), 'y': (0, 800)}

DLC_PATH = MAIN_PATH + '/pose_estimation/deeplabcut/projects/pogona_pursuit_realtime'
DLC_CONFIG_FILE = DLC_PATH + '/config.yaml'
ITERATION = 3
EXPORTED_MODEL_PATH = DLC_PATH + f'/exported-models/DLC_pogona_pursuit_resnet_50_iteration-{ITERATION}_shuffle-1'
PROBABILITY_THRESH = 0.85
BODY_PARTS = ['nose', 'left_ear', 'right_ear']