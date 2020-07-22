from flask_caching import Cache
from enum import Enum


class CacheColumns(Enum):
    """
    Enum for the Cache Columns
    name = (type, TTL)
    """
    EXPERIMENT_NAME = (str, 60 * 60)
    EXPERIMENT_PATH = (str, 60 * 60)
    MANUAL_RECORD_STOP = (bool, 5)


config = {
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_HOST': 'redis',
}


class RedisCache(Cache):
    def get(self, cache_column: Enum, *args, **kwargs):
        return super().get(cache_column.name, *args, **kwargs)

    def set(self, cache_column: Enum, value, *args, **kwargs):
        assert isinstance(value, cache_column.value[0]), \
            f'Bad type for {cache_column.name}; received {type(value)} expected {cache_column.value[0]}'
        if not kwargs.get('timeout') and cache_column.value:
            kwargs['timeout'] = cache_column.value[1]
        return super().set(cache_column.name, value, **kwargs)

    def delete(self, cache_column: Enum, *args, **kwargs):
        return super().delete(cache_column.name, *args, **kwargs)


def get_cache(app):
    return RedisCache(app, config=config)
