import re
import logging
from pathlib import Path
import subprocess

CRON_DIR = '/data/Pogona_Pursuit/cron'
ORIGIN = '/data/Pogona_Pursuit/Arena/experiments'
TARGET = '/media/sil2/regev/Pogona_Pursuit/Arena/experiments'
CACHE_FILE = Path(f'{CRON_DIR}/sil2_cache.txt')
TMP_DIR = '/tmp/experiments'
logger = logging.getLogger(__file__)
fh = logging.FileHandler(f'{CRON_DIR}/output.log')
fh.setFormatter(logging.Formatter(f'%(asctime)s - %(message)s'))
logger.addHandler(fh)


def load_cache():
    if not CACHE_FILE.exists():
        with CACHE_FILE.open('w') as f:
            f.write('')
    with CACHE_FILE.open('r') as f:
        cached = f.read()

    return cached.split()


def add_to_cache(exp_dir):
    with CACHE_FILE.open('a') as f:
        f.write(exp_dir)


def main(origin, target):
    logger.info('Start backup of experiments')
    cached = load_cache()
    experiments = Path(origin).glob('*')
    for exp_dir in experiments:
        try:
            if not re.match(r'\w+_\d{8}T\d{6}', exp_dir.name) or exp_dir.name in cached:
                continue

            tmp_exp = f'{TMP_DIR}/{exp_dir.name}'
            subprocess.run(['mkdir', '-p', tmp_exp])
            subprocess.run(['cp', '-r', exp_dir.as_posix(), tmp_exp])
            for video_path in Path(tmp_exp).glob('**/*.avi'):
                try:
                    vid_tmp = video_path.absolute().as_posix()
                    subprocess.run(['ffmpeg', '-i', vid_tmp, '-vcodec', 'libx265',
                                    '-crf', '28', vid_tmp.replace('.avi', '.mp4')])
                    subprocess.run(['rm', '-f', vid_tmp])
                except Exception as exc:
                    logger.error(f'Error converting video: {video_path.name}')

            subprocess.run(['cp', '-r', tmp_exp, target])
            add_to_cache(exp_dir.name)
            logger.info(f'{exp_dir.name} successfully copied to sil2')
            break

        except Exception as exc:
            logger.error(f'Error with {exp_dir}; {exc}')


if __name__ == "__main__":
    main(ORIGIN, TARGET)
