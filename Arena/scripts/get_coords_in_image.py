import matplotlib.pyplot as plt
import cv2
import yaml
from pathlib import Path
import numpy as np

IMAGE_PATH = '/data/Pogona_Pursuit/output/strike_analysis/PV80/6980/1856.jpg'


def onclick(event):
    print(f'[{round(event.xdata)},{round(event.ydata)}]', end=',')
    # print("button=%d, x=%d, y=%d, xdata=%f, ydata=%f" % (
    #      event.button, event.x, event.y, event.xdata, event.ydata))


def is_in_screen():
    s = yaml.load(Path('/data/Pogona_Pursuit/Arena/analysis/screen_coords.yaml').open(), Loader=yaml.FullLoader)
    cnt = s['screens']['pogona_pursuit2']
    cnt = np.array(cnt)

    img = cv2.imread(IMAGE_PATH)
    ax = plt.imshow(img)
    fig = ax.get_figure()

    def onclick_(event):
        result = cv2.pointPolygonTest(cnt, (event.xdata, event.ydata), False)
        text = 'outside'
        if result == 1:
            text = 'inside'
        elif result == 0:
            text = 'on contour'
        print(f'[{round(event.xdata)},{round(event.ydata)}] - {text}')

    cid = fig.canvas.mpl_connect('button_press_event', onclick_)
    plt.show()


def select_coords():
    img = cv2.imread(IMAGE_PATH)
    ax = plt.imshow(img)
    fig = ax.get_figure()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


if __name__ == '__main__':
    is_in_screen()
