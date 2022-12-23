import os
os.chdir('..')
import cv2
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from image_handlers.pogona_head import YOLOv5Detector
from utils import KalmanFilter
from db_models import ORM, Video


def to_centroid(xA, yA, xB, yB):
    return (xA + xB) / 2, (yA + yB) / 2


def predict(yolo, video_path):
    kalman = KalmanFilter()
    cap = cv2.VideoCapture(video_path)
    frames_times = pd.read_csv(Path(video_path).with_suffix('.csv')).timestamp.values
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    res = []
    for i in tqdm(range(n_frames), desc=video_path):
        ret, frame = cap.read()
        det, _ = yolo.detect_image(frame)
        if det is not None:
            x, y = to_centroid(*det[:4])
            x, y = kalman.get_filtered((x, y))
        else:
            x, y = None, None
        res.append((frames_times[i], x, y))
    return res


def main():
    orm = ORM()
    yolo = YOLOv5Detector(return_neareast_detection=False)
    yolo.load()
    with orm.session() as s:
        for v in s.query(Video).filter_by(animal_id='PV87', cam_name='top').all():
            if v.predictions is not None:
                continue
            res = predict(yolo, v.path)
            orm.commit_video_predictions('pogona_head_local', res, v.id, v.start_time)


if __name__ == '__main__':
    main()
