import redis
from enum import Enum

REDIS_HOST = 'cache'


class CacheColumns(Enum):
    """
    Enum for the Cache Columns
    name = (type, TTL)
    """
    EXPERIMENT_NAME = (str, 60 * 60)
    EXPERIMENT_PATH = (str, 60 * 60)
    EXPERIMENT_TRIAL_PATH = (str, 60 * 60)
    STREAM_CAMERA = (str, 60)
    MANUAL_RECORD_STOP = (bool, 5)


class RedisCache:
    def __init__(self):
        self._redis = redis.Redis(host=REDIS_HOST, port=6379, db=0)

    def get(self, cache_column: Enum):
        res = self._redis.get(cache_column.name)
        if res and type(res) == bytes:
            return res.decode("utf-8")
        return res

    def set(self, cache_column: Enum, value, timeout=None):
        assert isinstance(value, cache_column.value[0]), \
            f'Bad type for {cache_column.name}; received {type(value)} expected {cache_column.value[0]}'
        if not timeout and cache_column.value:
            timeout = cache_column.value[1]
        return self._redis.set(cache_column.name, value, ex=timeout)

    def delete(self, cache_column: Enum):
        return self._redis.delete(cache_column.name)
