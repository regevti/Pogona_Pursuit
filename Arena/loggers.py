import sys
import time
import logging
import logging.handlers
import logging.config
import traceback
import multiprocessing as mp
import threading


def logger_thread(q: mp.Queue, stop_event: mp.Event):
    def _logger_thread():
        logger = logging.getLogger('logger_thread')
        logger.setLevel(logging.DEBUG)
        h = logging.StreamHandler()
        h.setFormatter(CustomFormatter())
        logger.addHandler(h)

        while not stop_event.is_set():
            record = q.get()
            logger.handle(record)

    t = threading.Thread(target=_logger_thread)
    t.start()
    return t


def get_process_logger(name, q: mp.Queue):
    qh = logging.handlers.QueueHandler(q)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(qh)
    return logger


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setLevel(logging.DEBUG)
        h.setFormatter(CustomFormatter())
        logger.addHandler(h)
    return logger


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = '%(asctime)s %(name)-15s %(levelname)-8s %(processName)-12s - %(message)s'

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, '%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


# deprecated
class MultiProcessingLog(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)

        # self._handler = RotatingFileHandler('arena')
        self._handler = logging.StreamHandler()
        self.setFormatter(CustomFormatter())
        self.queue = mp.Queue(-1)

        t = threading.Thread(target=self.receive)
        t.daemon = True
        t.start()

    def setFormatter(self, fmt):
        logging.Handler.setFormatter(self, fmt)
        self._handler.setFormatter(fmt)

    def receive(self):
        while True:
            try:
                record = self.queue.get()
                self._handler.emit(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except:
                traceback.print_exc(file=sys.stderr)

    def send(self, s):
        self.queue.put_nowait(s)

    def _format_record(self, record):
        # ensure that exc_info and args have been stringified.  Removes any chance of unpickleable things inside and
        # possibly reduces message size sent over the pipe
        if record.args:
            record.msg = record.msg % record.args
            record.args = None
        if record.exc_info:
            dummy = self.format(record)
            record.exc_info = None

        return record

    def emit(self, record):
        try:
            s = self._format_record(record)
            self.send(s)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

    def close(self):
        self._handler.close()
        logging.Handler.close(self)
