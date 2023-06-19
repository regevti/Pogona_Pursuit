import time
import yaml
import json
from datetime import datetime, timedelta
from pathlib import Path

import utils
from db_models import ORM, Block, Experiment
from cache import RedisCache, CacheColumns as cc
from experiment import ExperimentCache, ExperimentValidation
from loggers import get_logger
import config

CONFIG_PATH = 'configurations/agent_config.yaml'
EXIT_HOLES = ['bottomLeft', 'bottomRight']


class Agent:
    def __init__(self, orm=None):
        self.orm = orm if orm is not None else ORM()
        self.logger = get_logger('Agent')
        self.cache = RedisCache()
        self.exp_validation = ExperimentValidation(logger=self.logger, orm=self.orm, cache=self.cache, is_silent=True)
        agent_config = yaml.load(Path(CONFIG_PATH).open(), Loader=yaml.FullLoader)
        self.check_agent_config(agent_config)
        self.animal_id = None
        self.trials = agent_config['trials']
        self.default_struct = agent_config['default_struct']
        self.times = agent_config['times']
        self.history = {}
        self.next_trial_name = None

    def update(self):
        self.animal_id = self.cache.get(cc.CURRENT_ANIMAL_ID)
        self.init_history()
        self.load_history()
        self.next_trial_name = self.get_next_trial_name()
        self.create_cached_experiment()
        if not self.next_trial_name:
            # all experiments are over
            return

        if self.exp_validation.is_ready():
            self.schedule_next_block()
        else:
            error_msg = f'Unable to schedule an experiment using agent since the following checks failed: ' \
                        f'{",".join(self.exp_validation.failed_checks)}'
            self.publish(error_msg)

    def schedule_next_block(self):
        next_schedules = self.get_upcoming_agent_schedules()
        if next_schedules:
            # if there are scheduled agent trials, do nothing
            return

        possible_times = self.get_possible_times()
        if not possible_times:
            return
        self.orm.commit_schedule(possible_times[0], self.cached_experiment_name)

    def get_next_trial_name(self):
        for trial_name in self.trials:
            if self.is_trial_type_finished(trial_name):
                continue
            else:
                return trial_name

    def get_upcoming_agent_schedules(self):
        return {s.experiment_name: s.date for s in self.orm.get_upcoming_schedules().all()}

    def get_possible_times(self):
        now = datetime.now()
        start_hour, start_minute = self.times['start_time'].split(':')
        dt = now.replace(hour=int(start_hour), minute=int(start_minute), second=0)
        end_hour, end_minute = self.times['end_time'].split(':')
        end_dt = now.replace(hour=int(end_hour), minute=int(end_minute), second=0)
        possible_times = []
        while dt <= end_dt:
            if dt >= now:
                possible_times.append(dt)
            dt += timedelta(minutes=self.times['time_between_blocks'])
        return possible_times

    def init_history(self):
        self.history = {}
        for trial_name, trial_dict in self.trials.items():
            self.history[trial_name] = {'key': trial_dict['count']['key']}
            if 'per' in trial_dict['count']:
                self.history[trial_name]['counts'] = {}
                for group_name, group_vals in trial_dict['count']['per'].items():
                    self.history[trial_name]['counts'][group_name] = {k: 0 for k in group_vals}
            else:
                self.history[trial_name]['counts'] = 0

    def load_history(self):
        with self.orm.session() as s:
            exps = s.query(Experiment).filter_by(animal_id=self.animal_id).all()
            for exp in exps:
                for blk in exp.blocks:
                    if blk.movement_type in self.trials:
                        count_key = self.history[blk.movement_type]['key']
                        counts = self.history[blk.movement_type]['counts']
                        if isinstance(counts, dict):  # case of per
                            for metric_name, metric_counts in counts.items():
                                if getattr(blk, metric_name) in metric_counts:
                                    metric_counts[getattr(blk, metric_name)] += len(getattr(blk, count_key))
                        elif isinstance(counts, int):
                            self.history[blk.movement_type]['counts'] += len(getattr(blk, count_key))

    def publish(self, msg):
        last_publish = self.cache.get(cc.LAST_TIME_AGENT_MESSAGE)
        if not last_publish or time.time() - float(last_publish) > config.AGENT_MIN_DURATION_BETWEEN_PUBLISH:
            self.logger.error(msg)
            utils.send_telegram_message(f'Agent Message:\n{msg}')
            self.cache.set(cc.LAST_TIME_AGENT_MESSAGE, time.time())

    def get_animal_history(self):
        txt = f'Animal ID: {self.animal_id}\n'
        for trial_name in self.history:
            self.history[trial_name]['is_finished'] = self.is_trial_type_finished(trial_name)
        txt += json.dumps(self.history, indent=4)
        return txt

    def create_cached_experiment(self):
        # load the agent config
        block_dict_ = self.trials[self.next_trial_name].copy()
        count_dict = block_dict_.pop('count')
        for k, v in block_dict_.copy().items():
            if isinstance(v, str) and v.startswith('per_'):
                per_left = [x for x in count_dict['per'][k]
                            if self.history[self.next_trial_name]['counts'][k][x] < count_dict['amount']]
                if v == 'per_random':
                    block_dict_[k] = per_left
                elif v == 'per_ordered':
                    block_dict_[k] = per_left[0]

        json_struct = self.default_struct.copy()
        for k in ['exit_hole', 'reward_any_touch_prob']:
            if k in block_dict_:
                json_struct[k] = block_dict_.pop(k)
        json_struct['blocks'][0].update(block_dict_)
        json_struct['bug_types'] = self.get_bug_types()
        exp_name = self.save_cached_experiment(json_struct)
        return exp_name

    def save_cached_experiment(self, trial_dict):
        trial_dict['name'] = self.cached_experiment_name
        ExperimentCache().save(trial_dict)
        return trial_dict['name']

    @property
    def cached_experiment_name(self):
        return f'agent_{self.next_trial_name}'

    def is_trial_type_finished(self, trial_name):
        count_dict = self.trials[trial_name]['count']
        is_finished = False
        history = self.history[trial_name]['counts']
        if isinstance(history, dict):
            is_finished = all(v >= count_dict['amount'] for group_name, group_vals in history.items()
                              for v in group_vals.values())
        elif isinstance(history, (int, float)):
            is_finished = history >= count_dict['amount']
        return is_finished

    def get_bug_types(self):
        try:
            d = self.orm.get_animal_settings(self.animal_id)
        except Exception:
            d = {}
        return d.get('bug_types', ['cockroach'])

    def check_agent_config(self, agent_config):
        main_keys = ['trials', 'default_struct', 'times']
        for k in main_keys:
            assert k in agent_config, f'{k} must be in agent config'

        times_keys = ['start_time', 'end_time', 'time_between_blocks']
        for k in times_keys:
            assert k in agent_config['times'], f'{k} must be in times'

        for trial_name, trial_dict in agent_config['trials'].items():
            assert 'count' in trial_dict, f'"count" must be in trial {trial_name}'
            for k in ['key', 'amount']:
                assert k in trial_dict['count'], f'{k} must be in "count" of trial {trial_name}'


if __name__ == '__main__':
    ag = Agent()
    ag.update()
    print(ag.get_animal_history())
