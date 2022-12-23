from pathlib import Path
import pandas as pd
import json
import re
import os
from datetime import datetime, timedelta

os.chdir('../Arena')
from db_models import ORM, Experiment, Block

ROOT_DIR = '/media/reptilearn_lab/sessions'
SESSION_START = "session/run"
SESSION_STOP = "session/stop"
TRIAL_CHANGED = "('session', 'cur_trial')"
BLOCK_CHANGED = "('session', 'cur_block')"
session_events = [TRIAL_CHANGED, BLOCK_CHANGED, SESSION_START, SESSION_STOP]
orm = ORM()


class A:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def commit_block(start_time, bc, block_id):
    start_time = datetime.fromtimestamp(start_time)
    bid = orm.commit_block(A(start_time=start_time, end_time=None, block_id=block_id + 1, num_trials=bc['$num_trials'],
                              trial_duration=bc.get('$trial_duration'), iti=bc['$inter_trial_interval'], block_type='bugs',
                              bug_types=bc['bugTypes'], bug_speed=bc['speed'], bug_size=bc['bugSize'], is_default_bug_size=True,
                              exit_hole=bc['targetDrift'], reward_type='always', reward_bugs=bc['rewardBugs'],
                              reward_any_touch_prob=1, media_url='', movement_type=bc['movementType'],
                              is_anticlockwise=bc['isAntiClockWise'], target_drift=bc['targetDrift'], bug_height=bc['bugHeight'],
                              time_between_bugs=bc['timeBetweenBugs'], background_color=bc['backgroundColor']), is_cache_set=False)
    return bid


def update_block(bid, end_time, videos, animal_id):
    orm.update_block_end_time(block_id=bid, end_time=datetime.fromtimestamp(end_time))
    with orm.session() as s:
        blk = s.query(Block).filter_by(id=bid).first()
    for v in videos:
        cam_name, time = v.stem.split('_')
        time = datetime.strptime(time, '%Y%m%d-%H%M%S')
        if time >= blk.start_time and time < blk.end_time:
            frames = pd.read_csv(Path(v).with_suffix('.csv'))
            calc_fps = 1 / frames.timestamp.diff().mean()
            vid = orm.commit_video(str(v), round(calc_fps), cam_name, time, animal_id, bid)
            orm.commit_video_frames(frames.timestamp.values, vid)


def commit_trial(start_time, end_time, i, bug_trajectory, bid):
    return orm.commit_trial({
        'in_block_trial_id': i + 1,
        'start_time': datetime.fromtimestamp(start_time),
        'bug_trajectory': bug_trajectory.query(f'time>{start_time*1000} and time<{end_time*1000}').to_dict(orient='records'),
        'block_id': bid,
        'end_time': datetime.fromtimestamp(end_time),
        'duration': end_time - start_time
    })


def commit_strike(tid, trial_id, data, bid):
    data.update({'is_climbing': False, 'in_block_trial_id': trial_id, 'trial_id': tid, 'block_id': bid})
    data['time'] = datetime.fromtimestamp(data['time'] / 1000)
    orm.commit_strike(data)


def main():
    BAD_RECS = [
        'PV87_EP_hunter_trial5_20221031_100512',  # stop in middle of session and revert
        'PV87_EP_trail1_20221026_135259'  # stop in middle of session and revert
    ]

    for p in Path(ROOT_DIR).glob('*'):
        if not p.is_dir() or p.name.startswith('.') or p.name in BAD_RECS:
            continue

        m = re.search(r'(?P<animal_id>PV\d+)_((?P<exp_name>\w+)_)?(?P<date>\d{8})_(?P<hour>\d{6})', p.name)
        if not m:
            continue
        animal_id = m.group('animal_id')
        if animal_id not in ['PV87'] or int(m.group('date')) < 20220907:
            continue

        events_path = p / 'events.csv'
        exp_name = m.group('exp_name') or f"EXP{m.group('date')}"
        if not events_path.exists():
            continue
        try:
            events_df = pd.read_csv(events_path)
            if events_df.empty:
                continue

            sf = events_df.query(f'event=="{SESSION_START}"')
            if sf.empty:
                continue
            conf = json.loads(sf.iloc[0]['value'])
            app_params = conf.get('params', {})
            n_blocks = len(conf['blocks'])
            start_time = datetime.fromisoformat(conf['start_time'])
            end_experiment_time = datetime.fromtimestamp(events_df.query(f'event=="{SESSION_STOP}"').iloc[0].time)
            if (p / 'bug_trajectory.csv').exists():
                bug_trajectory = pd.read_csv(p / 'bug_trajectory.csv')
            else:
                bug_trajectory = pd.DataFrame(columns=['time'])
            strikes_df = events_df.query('event=="screen_touch"').drop(columns=['event']).copy()
            event_time_col = strikes_df.rename(columns={'time': 'event_time'})['event_time']
            strikes_df = strikes_df.value.apply(lambda x: pd.Series(json.loads(x)))
            strikes_df = pd.concat([event_time_col, strikes_df], axis=1)
            videos = list(p.glob('*.mp4'))
            with orm.session() as s:
                exp = s.query(Experiment).filter_by(name=exp_name).first()
                if exp:
                    orm.current_experiment_id = exp.id
                    # print(f'skipping {p}; already committed')
                    # continue
                else:
                    orm.commit_experiment(A(name=exp_name, start_time=start_time, end_time=end_experiment_time,
                                            animal_id=animal_id, cameras={}, num_blocks=n_blocks,
                                            extra_time_recording=0, time_between_blocks=0, experiment_path=str(p.resolve())))

                sess_events = events_df.query(f'event in {session_events}').reset_index(drop=True)
                block_id, trial_id = 0, 0
                # commit blocks and trials
                for i, row in sess_events.iterrows():
                    bc = app_params.copy()
                    bc.update(conf['blocks'][block_id])
                    if not bc:
                        continue
                    start_time = row.time
                    if row.event == SESSION_STOP:
                        orm.update_experiment_end_time(datetime.fromtimestamp(start_time))
                        update_block(bid, start_time, videos, animal_id)
                    else:
                        end_time = sess_events.iloc[i+1]['time']
                        if row.event == SESSION_START:
                            bid = commit_block(start_time, bc, block_id)
                            tid = commit_trial(start_time, end_time, i, bug_trajectory, bid)
                            for j, strk_row in strikes_df.query(f'event_time >= {start_time} and event_time <= {end_time}').drop(columns=['event_time']).iterrows():
                                commit_strike(tid, trial_id+1, strk_row.to_dict(), bid)
                        elif row.event == BLOCK_CHANGED:
                            block_id += 1
                            update_block(bid, start_time, videos, animal_id)
                            bid = commit_block(start_time, bc, block_id)
                        elif row.event == TRIAL_CHANGED:
                            trial_id += 1
                            tid = commit_trial(start_time, end_time, i, bug_trajectory, bid)
                            for j, strk_row in strikes_df.query(f'event_time >= {start_time} and event_time <= {end_time}').drop(columns=['event_time']).iterrows():
                                commit_strike(tid, trial_id+1, strk_row.to_dict(), bid)
                        else:  # end session
                            pass

                s.commit()
                print(f'finished {p.name}; experiment_id = {exp.id if exp is not None else None}')

        except ImportError as exc:
            print(f'Error parsing {events_path}; {exc}')


if __name__ == '__main__':
    main()