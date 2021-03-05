import redis
import config
EXPERIMENTS_TIMEOUT = 60 * 60


class Column:
    def __init__(self, name, typ, timeout):
        self.name = name
        self.type = typ
        self.timeout = timeout


class CacheColumns:
    """Cache Columns used by RedisCache"""
    EXPERIMENT_NAME = Column('EXPERIMENT_NAME', str, EXPERIMENTS_TIMEOUT)
    EXPERIMENT_PATH = Column('EXPERIMENT_PATH', str, EXPERIMENTS_TIMEOUT)
    EXPERIMENT_BLOCK_PATH = Column('EXPERIMENT_BLOCK_PATH', str, EXPERIMENTS_TIMEOUT)
    EXPERIMENT_BLOCK_ON = Column('EXPERIMENT_BLOCK_ON', bool, EXPERIMENTS_TIMEOUT)
    APP_ON = Column('APP_ON', bool, EXPERIMENTS_TIMEOUT)
    ALWAYS_REWARD = Column('ALWAYS_REWARD', bool, EXPERIMENTS_TIMEOUT)
    STREAM_CAMERA = Column('STREAM_CAMERA', str, 60)
    MANUAL_RECORD_STOP = Column('MANUAL_RECORD_STOP', bool, 5)
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
