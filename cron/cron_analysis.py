import deeplabcut
import re
from pathlib import Path

HOME_DIR = '/media/sil2/regev/Pogona_Pursuit/Arena/experiments'
# HOME_DIR = '/media/sil2/regev/pose_estimation/deeplabcut/projects/pogona_pursuit-regev-2020-07-19/test_videos/'
path_config_file = '/media/sil2/regev/pose_estimation/deeplabcut/projects/pogona_pursuit-regev-2020-07-19/config.yaml'

for video_file in Path(HOME_DIR).glob('**/*.avi'):
    try:
        if video_file.name.startswith('delete'):
            continue
        h5_file = list(video_file.parent.rglob(f'{video_file.stem}*.h5'))
        if h5_file and len(h5_file) == 1:
            continue

        deeplabcut.analyze_videos(path_config_file,[video_file.as_posix()], videotype='.avi', gputouse=0, save_as_csv=True)
    except Exception as exc:
        print(f'ERROR in file {video_file.as_posix()}\n{exc}')