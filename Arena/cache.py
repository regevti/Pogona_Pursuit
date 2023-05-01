import redis
import config


class Column:
    def __init__(self, name, typ, timeout):
        self.name = name
        self.type = typ
        self.timeout = timeout


class CacheColumns:
    """
    Cache Columns used by RedisCache
    set "static" for timeout to prevent deleting of column after arena init
    """
    EXPERIMENT_NAME = Column('EXPERIMENT_NAME', str, config.experiments_timeout)
    EXPERIMENT_PATH = Column('EXPERIMENT_PATH', str, config.experiments_timeout)
    EXPERIMENT_BLOCK_ID = Column('EXPERIMENT_BLOCK_ID', int, config.experiments_timeout)
    EXPERIMENT_BLOCK_PATH = Column('EXPERIMENT_BLOCK_PATH', str, config.experiments_timeout)
    STREAM_CAMERA = Column('STREAM_CAMERA', str, config.experiments_timeout)
    IS_RECORDING = Column('IS_RECORDING', bool, None)
    IS_VISUAL_APP_ON = Column('IS_VISUAL_APP_ON', bool, config.experiments_timeout)
    IS_ALWAYS_REWARD = Column('IS_ALWAYS_REWARD', bool, config.experiments_timeout)
    IS_REWARD_TIMEOUT = Column('IS_REWARD_TIMEOUT', bool, config.reward_timeout)
    ACTIVE_CAMERAS = Column('ACTIVE_CAMERAS', list, config.experiments_timeout)
    RECORDING_CAMERAS = Column('RECORDING_CAMERAS', list, config.experiments_timeout)
    CURRENT_BLOCK_DB_INDEX = Column('CURRENT_BLOCK_DB_INDEX', int, config.experiments_timeout)
    OPEN_APP_HOST = Column('OPEN_APP_HOST', str, 60)
    CURRENT_ANIMAL_ID = Column('CURRENT_ANIMAL_ID', str, 'static')
    CURRENT_ANIMAL_ID_DB_INDEX = Column('CURRENT_ANIMAL_ID_DB_INDEX', int, 'static')
    REWARD_LEFT = Column('REWARD_LEFT', list, 'static')
    IS_EXPERIMENT_CONTROL_CAMERAS = Column('IS_EXPERIMENT_CONTROL_CAMERAS', bool, config.experiments_timeout)
    CAM_TRIGGER_STATE = Column('CAM_TRIGGER_STATE', int, None)
    CAM_TRIGGER_DISABLE = Column('CAM_TRIGGER_DISABLE', bool, config.experiments_timeout)


class RedisCache:
    def __init__(self):
        self._redis = redis.Redis(host=config.redis_host, port=6379, db=0)

    def get(self, cache_column: Column):
        res = self._redis.get(cache_column.name)
        if res:
            if type(res) == bytes:
                res = res.decode("utf-8")

            if cache_column.type == bool:
                res = int(res)
            elif cache_column.type == list:
                res = res.split(',')
        return res

    def set(self, cache_column: Column, value, timeout=None):
        assert isinstance(value, cache_column.type), \
            f'Bad type for {cache_column.name}; received {type(value)} expected {cache_column.type}'

        if not timeout and cache_column.timeout and cache_column.timeout != 'static':
            timeout = cache_column.timeout

        if cache_column.type == bool:
            value = int(value)
        elif cache_column.type == list:
            value = ','.join(value)
        return self._redis.set(cache_column.name, value, ex=timeout)

    def update_cam_dict(self, cam_name, **kwargs):
        key = self._get_cam_dict_key(cam_name)
        d = self.get_cam_dict(cam_name)
        d.update(kwargs)
        for k, v in d.copy().items():
            if isinstance(v, dict):
                d.pop(k)
            if isinstance(v, (list, tuple)):
                d[f'{k}_list'] = ','.join([str(x) for x in v])
                d.pop(k)
            if v is None:  # if none
                d[k] = ''
        return self._redis.hmset(key, d)

    def get_cam_dict(self, cam_name):
        d = self._redis.hgetall(self._get_cam_dict_key(cam_name))
        for k, v in d.copy().items():
            if type(v) == bytes:
                v = v.decode("utf-8")
            if type(k) == bytes:
                d.pop(k)  # remove binary keys from dict
                k = k.decode("utf-8")

            if k.endswith('_list'):
                k, v = k.replace('_list', ''), v.split(',')
                if k in ['image_size']:
                    v = [int(x) for x in v]

            d[k] = v
        return d

    def delete_cam_dict(self, cam_name):
        self._redis.delete(self._get_cam_dict_key(cam_name))

    def set_cam_output_dir(self, cam_name, output_dir: str):
        self.update_cam_dict(cam_name, **{config.output_dir_key: output_dir})

    def _get_cam_dict_key(self, cam_name):
        return f'cam_{cam_name}'

    def append_to_list(self, cache_column: Column, value, timeout=None):
        assert cache_column.type == list
        l = self.get(cache_column) or []
        l.append(value)
        return self.set(cache_column, l, timeout)

    def remove_from_list(self, cache_column: Column, value):
        assert cache_column.type == list
        l = self.get(cache_column) or []
        if value in l:
            l.remove(value)
        return self.set(cache_column, l)

    def delete(self, cache_column: Column):
        return self._redis.delete(cache_column.name)

    def publish(self, channel, payload=''):
        self._redis.publish(channel, payload)

    def publish_command(self, command, payload=''):
        assert command in config.commands_topics, f'command {command} is not in config commands_topics'
        self.publish(config.commands_topics[command], payload)

    def get_current_experiment(self):
        return self.get(CacheColumns.EXPERIMENT_NAME)

    def stop_experiment(self):
        self.delete(CacheColumns.EXPERIMENT_NAME)
