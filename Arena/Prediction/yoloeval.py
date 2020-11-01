import os
os.chdir('/app/Pogona_realtime/Arena')

import cv2 as cv
import numpy as np
from glob import glob
from tqdm.auto import tqdm
import pandas as pd
from time import time
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import json

from Prediction import detector
from Prediction import dataset
from Prediction import train_eval

output_dir = '../yolo_eval_frames'
im_paths = sorted(glob(os.path.join(output_dir, "*.jpg")))
det = detector.Detector_v4(conf_thres=0.8, nms_thres=0.5)
det_df = pd.DataFrame(columns=['filename', 'run_time_ms', 'x1', 'y1', 'x2', 'y2', 'confidence'])
i = 0

for im_path in tqdm(im_paths):
    im = cv.imread(im_path)
    start_time = time()
    ds = det.detect_image(im)
    run_time = (time() - start_time) * 1000
    if ds is not None:
        for d in ds:
            det_df.loc[i] = [im_path, run_time] + d.tolist()
    else:
        det_df.loc[i] = [im_path, run_time] + [np.nan]*5
    i+=1
    
sns.distplot(det_df['run_time_ms'], bins=100, kde=False)
plt.xlabel('millisecond')
plt.savefig('../experiments_plots/yolo_timings.png', dpi=200)