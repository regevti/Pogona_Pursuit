import redis
import config


class Column:
    def __init__(self, name, typ, timeout):
        self.name = name
        self.type = typ
        self.timeout = timeout


class CacheColumns:
    """Cache Columns used by RedisCache"""
    EXPERIMENT_NAME = Column('EXPERIMENT_NAME', str, config.experiments_timeout)
    EXPERIMENT_PATH = Column('EXPERIMENT_PATH', str, config.experiments_timeout)
    EXPERIMENT_BLOCK_ID = Column('EXPERIMENT_BLOCK_ID', int, config.experiments_timeout)
    EXPERIMENT_BLOCK_PATH = Column('EXPERIMENT_BLOCK_PATH', str, config.experiments_timeout)
    STREAM_CAMERA = Column('STREAM_CAMERA', str, 60)
    IS_RECORDING = Column('IS_RECORDING', bool, None)
    IS_VISUAL_APP_ON = Column('IS_VISUAL_APP_ON', bool, config.experiments_timeout)
    IS_ALWAYS_REWARD = Column('IS_ALWAYS_REWARD', bool, config.experiments_timeout)
    IS_REWARD_TIMEOUT = Column('IS_REWARD_TIMEOUT', bool, 30)


class RedisCache:
    def __init__(self):
        self._redis = redis.Redis(host=config.redis_host, port=6379, db=0)

    def get(self, cache_column: Column):
        res = self._redis.get(cache_column.name)
        if res and type(res) == bytes:
            decoded = res.decode("utf-8")
            if cache_column.type == bool:
                decoded = int(decoded)
            return decoded
        return res

    def set(self, cache_column: Column, value, timeout=None):
        assert isinstance(value, cache_column.type), \
            f'Bad type for {cache_column.name}; received {type(value)} expected {cache_column.type}'
        if not timeout and cache_column.timeout:
            timeout = cache_column.timeout
        if cache_column.type == bool:
            value = int(value)
        return self._redis.set(cache_column.name, value, ex=timeout)

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
