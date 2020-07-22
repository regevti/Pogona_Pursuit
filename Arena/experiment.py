from utils import get_datetime_string, mkdir
from arena import OUTPUT_DIR, record
from cache import CacheColumns
from mqtt import MQTTClient
import time

mqtt_client = MQTTClient()


class Experiment:
    def __init__(self, name, cache, cameras, trial_duration=60, trials=1, iti=10):
        self.experiment_name = f'{name}_{get_datetime_string()}'
        self.cache = cache
        self.num_trials = trials
        self.trial_duration = trial_duration
        self.iti = iti
        self.cameras = cameras
        self.start()

    def start(self):
        mkdir(f'{OUTPUT_DIR}/{self.experiment_name}')
        self.cache.set(CacheColumns.EXPERIMENT_NAME, self.experiment_name, timeout=self.experiment_duration)
        self.cache.set(CacheColumns.EXPERIMENT_PATH, self.experiment_path, timeout=self.experiment_duration)
        mqtt_client.publish_command('hide_bugs')
        for i in range(self.num_trials):
            if i != 0:
                time.sleep(self.iti)
            self.run_trial()

    def run_trial(self):
        mqtt_client.publish_command('init_bugs', 1)
        record(cameras=self.cameras, output=self.videos_path, is_auto_start=True, record_time=self.trial_duration)
        mqtt_client.publish_command('hide_bugs')

    def stop(self):
        pass

    @property
    def experiment_duration(self):
        return self.num_trials * self.trial_duration + (self.num_trials - 1) * self.iti

    @property
    def experiment_path(self):
        return f'{OUTPUT_DIR}/{self.experiment_name}'

    @property
    def videos_path(self):
        return f'{self.experiment_path}/videos'
