import redis
import os
REDIS_HOST = os.environ.get('REDIS_HOST', 'cache')


class Column:
    def __init__(self, name, typ, timeout):
        self.name = name
        self.type = typ
        self.timeout = timeout


class CacheColumns:
    """Cache Columns used by RedisCache"""
    EXPERIMENT_NAME = Column('EXPERIMENT_NAME', str, 60 * 60)
    EXPERIMENT_PATH = Column('EXPERIMENT_PATH', str, 60 * 60)
    EXPERIMENT_TRIAL_PATH = Column('EXPERIMENT_TRIAL_PATH', str, 60 * 60)
    STREAM_CAMERA = Column('STREAM_CAMERA', str, 60)
    MANUAL_RECORD_STOP = Column('MANUAL_RECORD_STOP', bool, 5)


class RedisCache:
    def __init__(self):
        self._redis = redis.Redis(host=REDIS_HOST, port=6379, db=0)

    def get(self, cache_column: Column):
        res = self._redis.get(cache_column.name)
        if res and type(res) == bytes:
            return res.decode("utf-8")
        return res

    def set(self, cache_column: Column, value, timeout=None):
        assert isinstance(value, cache_column.type), \
            f'Bad type for {cache_column.name}; received {type(value)} expected {cache_column.type}'
        if not timeout and cache_column.timeout:
            timeout = cache_column.timeout
        return self._redis.set(cache_column.name, value, ex=timeout)

    def delete(self, cache_column: Column):
        return self._redis.delete(cache_column.name)
