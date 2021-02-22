import re
import logging
import shutil
from pathlib import Path
import subprocess

CRON_DIR = '/data/Pogona_Pursuit/cron'
ORIGIN = '/data/Pogona_Pursuit/Arena/experiments'
TARGET = '/media/sil2/regev/Pogona_Pursuit/Arena/experiments'
CACHE_FILE = Path(f'{CRON_DIR}/sil2_cache.txt')
TMP_DIR = '/tmp/experiments'
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
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


def main(origin, target_experiments_dir):
    logger.info('Start backup of experiments')
    cached = load_cache()
    subprocess.run(['mkdir', '-p', TMP_DIR])
    experiments = Path(origin).glob('*')
    for animal_dir in experiments:

        if animal_dir.name.startswith('delete'):
            shutil.rmtree(animal_dir.as_posix())

        for day_dir in animal_dir.glob('*'):
            if not re.match(r'\d+', day_dir.name):
                continue
            for block_dir in day_dir.glob('*'):
                path_id = "/".join(block_dir.parts[-3:])  # e.g. 9/20210220/block3
                try:
                    target = Path(target_experiments_dir) / path_id
                    if not re.match(r'block\d+', block_dir.name) or path_id in cached or target.exists():
                        continue

                    tmp_block_dir = Path(TMP_DIR) / path_id
                    tmp_block_dir.parent.mkdir(exist_ok=True, parents=True)
                    target.parent.mkdir(exist_ok=True, parents=True)

                    subprocess.run(['cp', '-r', block_dir.as_posix(), tmp_block_dir.parent.as_posix()])

                    for video_path in tmp_block_dir.glob('**/*.avi'):
                        try:
                            vid_tmp = video_path.absolute().as_posix()
                            subprocess.run(['ffmpeg', '-i', vid_tmp, '-c:v', 'libx265',
                                            '-preset', 'fast', '-crf', '28', '-tag:v', 'hvc1',
                                            '-c:a', 'eac3', '-b:a', '224k', vid_tmp.replace('.avi', '.mp4')])
                            subprocess.run(['rm', '-f', vid_tmp])
                        except Exception as exc:
                            logger.error(f'Error converting video: {video_path.name}; {exc}')

                    subprocess.run(['cp', '-r', tmp_block_dir.as_posix(), target.parent.as_posix()])
                    add_to_cache(path_id + '\n')
                    logger.info(f'{path_id} successfully copied to sil2')

                except Exception as exc:
                    logger.error(f'Error with {path_id}; {exc}')


if __name__ == "__main__":
    main(ORIGIN, TARGET)
