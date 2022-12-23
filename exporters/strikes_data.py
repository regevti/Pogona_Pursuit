from pathlib import Path
import pandas as pd
import json
import re
import os
from pathlib import Path
from datetime import datetime, timedelta

os.chdir('../Arena')
from db_models import ORM, Experiment, Block, Strike


def run(experiment_path):
    orm = ORM()
    d = []
    with orm.session() as s:
        exp = s.query(Experiment).filter_by(experiment_path=experiment_path).first()
        for blk in exp.blocks:
            for stk in blk.strikes:
                d.append({k: v for k, v in stk.__dict__.items() if not k.startswith('_')})
    return d


if __name__ == "__main__":
    d = run(experiment_path='/media/reptilearn_lab/sessions/PV87_EP_hunter_trial4_20221030_100000')
    df = pd.DataFrame(d)
    cache_dir = Path('/media/sil1/Pogona Vitticeps/PV87/PV87_EP_Hunter_session4') / 'regev_cache'
    cache_dir.mkdir(exist_ok=True)
    df.to_csv(cache_dir / 'behavior_timestamps.csv')

