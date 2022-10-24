import sys
import time
import logging
import logging.handlers
import logging.config
import traceback
import multiprocessing as mp
import threading

from cache import RedisCache, CacheColumns as cc
import config

cache = RedisCache()
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(processName)s - %(message)s'
DEFAULT_DATEFMT = '%Y-%m-%d %H:%M:%S'
_loggers = {}

log_conf = {
    'version': 1,
    'formatters': {
        'default': {
            'format': DEFAULT_FORMAT,
            'datefmt': DEFAULT_DATEFMT
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
        }
    },
    'root': {
        'level': config.LOGGING_LEVEL,
        'handlers': ['console'],
    }
}


def init_logger_config():
    logging.config.dictConfig(log_conf)
    root_logger = logging.getLogger()
    root_logger.info('Logging is configured')


_handlers = {}


def logger_thread(q: mp.Queue, stop_event: mp.Event):
    def _logger_thread():
        logger = get_logger('logger_thread')

        while not stop_event.is_set():
            record = q.get()
            logger.handle(record)

    t = threading.Thread(target=_logger_thread)
    t.start()
    return t


def get_process_logger(name, q: mp.Queue):
    if _handlers.get(name):
        qh = _handlers[name]
    else:
        qh = logging.handlers.QueueHandler(q)
        _handlers[name] = qh
    logger = logging.getLogger(name)
    logger.setLevel(logging.getLevelName(config.LOGGING_LEVEL))
    logger.addHandler(qh)
    logger.propagate = False
    return logger


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.getLevelName(config.LOGGING_LEVEL))
    if _handlers.get(name):
        h = _handlers[name]
    else:
        h = ArenaHandler()
        h.setFormatter(logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATEFMT))
        _handlers[name] = h
    logger.addHandler(h)
    logger.propagate = False
    _loggers[name] = logger
    return logger


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = DEFAULT_FORMAT

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, DEFAULT_DATEFMT)
        return formatter.format(record)


class ArenaHandler(logging.StreamHandler):
    """
    A handler class for publishing records through redis pub/sub
    """

    def emit(self, record):
        try:
            super().emit(record)
            msg = self.format(record)
            if record.levelno >= logging.getLevelName(config.UI_LOGGING_LEVEL):
                cache.publish(config.ui_console_channel, str(msg))
            block_path = cache.get(cc.EXPERIMENT_BLOCK_PATH)
            if block_path:
                with open(f'{block_path}/block.log', 'a') as f:
                    f.write(str(msg) + '\n')
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
