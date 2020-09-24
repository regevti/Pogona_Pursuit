import deeplabcut
import re
from pathlib import Path

HOME_DIR = Path('/media/sil2/regev/Pogona_Pursuit/Arena')
DLC_DIR = Path('/media/sil2/regev/pose_estimation/deeplabcut/projects')
path_config_file = DLC_DIR / '/pogona_pursuit-regev-2020-07-19/config.yaml'

for video_file in Path(HOME_DIR).glob('**/*.avi'):
    try:
        if video_file.name.startswith('delete'):
            continue
        h5_file = list(video_file.parent.rglob(f'{video_file.stem}*.h5'))
        if h5_file and len(h5_file) == 1:
            continue
        print(f'Start DLC analysis for video: {video_file}')
        deeplabcut.analyze_videos(path_config_file, [video_file.as_posix()], videotype='.avi', gputouse=0, save_as_csv=True)
    except Exception as exc:
        print(f'ERROR in file {video_file.as_posix()}\n{exc}')