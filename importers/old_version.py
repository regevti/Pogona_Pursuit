from pathlib import Path
import pandas as pd
import json
import re
import os
from datetime import datetime, timedelta

os.chdir('../Arena')
from db_models import ORM, Experiment, Block

ROOT_DIR = '/media/reptilearn_lab/sessions'
orm = ORM()


class A:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


if __name__ == '__main__':
    pass
