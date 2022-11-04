from pathlib import Path
import pandas as pd
import json

ROOT_DIR = '/media/reptilearn_lab/sessions'

if __name__ == '__main__':
    for p in Path(ROOT_DIR).glob('*'):
        if not p.is_dir() or p.name.startswith('.'):
            continue

        events_path = p / 'events.csv'
        if not events_path.exists():
            continue
        try:
            events_df = pd.read_csv(events_path).query('event=="screen_touch"').drop(columns=['event'])
            if events_df.empty:
                continue

            strikes_df = events_df.value.apply(lambda x: pd.Series(json.loads(x))).copy()
            n_touches = len(events_df)

            if n_touches > 0:
                print(f'{p.name}: {n_touches}')
        except Exception as exc:
            print(f'Error parsing {events_path}')
