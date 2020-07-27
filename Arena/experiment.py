from utils import get_datetime_string, mkdir
from arena import record
from cache import CacheColumns, RedisCache
from mqtt import MQTTClient
import time

mqtt_client = MQTTClient()
EXPERIMENTS_DIR = 'experiments'


class Experiment:
    def __init__(self, name: str, animal_id: str, cache: RedisCache, cameras, trial_duration=60, num_trials=1, iti=10):
        self.experiment_name = f'{name}_{get_datetime_string()}'
        self.animal_id = animal_id
        self.cache = cache
        self.num_trials = num_trials
        self.trial_duration = trial_duration
        self.iti = iti
        self.current_trial = 1
        self.cameras = cameras
        self.start()

    def __str__(self):
        output = ''
        for obj in ['experiment_name', 'animal_id', 'num_trials', 'cameras', 'trial_duration', 'iti']:
            output += f'{obj}: {getattr(self, obj)}\n'
        return output

    def start(self):
        mkdir(self.experiment_path)
        self.save_experiment_log()
        self.cache.set(CacheColumns.EXPERIMENT_NAME, self.experiment_name, timeout=self.experiment_duration)
        self.cache.set(CacheColumns.EXPERIMENT_PATH, self.experiment_path, timeout=self.experiment_duration)
        mqtt_client.publish_command('hide_bugs')
        for i in range(self.num_trials):
            if not self.cache.get(CacheColumns.EXPERIMENT_NAME):
                print('experiment was stopped')
                break
            self.current_trial = i + 1
            if i != 0:
                time.sleep(self.iti)
            self.run_trial()
        self.end_experiment()

    def run_trial(self):
        mkdir(self.trial_path)
        mqtt_client.publish_command('init_bugs', 1)
        self.cache.set(CacheColumns.EXPERIMENT_TRIAL_PATH, self.trial_path, timeout=self.trial_duration)
        record(cameras=self.cameras, output=self.videos_path, is_auto_start=True, record_time=self.trial_duration,
               cache=self.cache)
        mqtt_client.publish_command('hide_bugs')

    def end_experiment(self):
        self.cache.delete(CacheColumns.EXPERIMENT_NAME)
        self.cache.delete(CacheColumns.EXPERIMENT_PATH)

    def save_experiment_log(self):
        with open(f'{self.experiment_path}/experiment.log', 'w') as f:
            f.write(str(self))

    @property
    def experiment_duration(self):
        return round((self.num_trials * self.trial_duration + (self.num_trials - 1) * self.iti) * 1.5)

    @property
    def experiment_path(self):
        return f'{EXPERIMENTS_DIR}/{self.experiment_name}'

    @property
    def trial_path(self):
        return f'{self.experiment_path}/trial{self.current_trial}'

    @property
    def videos_path(self):
        return f'{self.trial_path}/videos'
