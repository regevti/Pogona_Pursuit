import cv2
import numpy as np
from scipy import optimize
from scipy.stats import zscore
import matplotlib.pyplot as plt
import colorsys
import matplotlib.collections as mcoll
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_rgb


def flatten(l):
    return [item for sublist in l if sublist for item in sublist]


def pixels2cm(x):
    return x * 0.01833304668870419


def closest_index(series, x, max_dist=0.050):
    diffs = (series - x).abs().dt.total_seconds()
    d = diffs[diffs <= max_dist]
    if len(d) > 0:
        return d.index[d.argmin()]


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def fit_circle(x, y):
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = np.mean(x), np.min(y)
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2 = calc_R(*center_2)
    R_2 = Ri_2.mean()
    #     residu_2 = sum((Ri_2 - R_2) ** 2)

    return xc_2, yc_2, R_2


def transform_circle_center(x, y, x_center, y_center):
    x1 = x - x_center
    y1 = -y - y_center

    return x1, y1


def polar_transform(x, y, x_center, y_center):
    theta = lambda x1, y1: np.arctan2(x1, y1)
    rho = lambda x1, y1: np.sqrt(x1 ** 2 + y1 ** 2)
    x1, y1 = transform_circle_center(x, y, x_center, y_center)
    return theta(x1, y1), rho(x1, y1)
    # return np.array([theta(x, y) - theta(cx, cy), rho(x, y) - rho(cx, cy)])


def project(cx, cy, x, y) -> np.ndarray:
    th = np.arctan2(cx, cy)
    th = th - np.pi / 2
    r = np.array(((np.cos(th), -np.sin(th)),
                  (np.sin(th), np.cos(th))))
    u = np.array([x, y]) - np.array([cx, cy])
    xr = r.dot(np.array([1, 0]))
    yr = r.dot(np.array([0, 1]))
    projection = lambda x1, y1: y1 * np.dot(y1, x1) / np.dot(y1, y1)
    v = np.array([np.dot(projection(u, xr), xr), np.dot(projection(u, yr), yr)])
    if np.abs(v[0]) > 1000 or np.abs(v[1]) > 1000:
        return np.array([np.nan, np.nan])
    return v


def colorline(ax, x, y, z=None, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0,
              set_ax_lim=True):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
    z = np.asarray(z)
    segments = make_segments(x, y)

    if not isinstance(cmap, LinearSegmentedColormap):
        rgb = to_rgb(cmap)
        h, s, v = colorsys.rgb_to_hsv(*rgb)
        cmap = ListedColormap([colorsys.hsv_to_rgb(h, s=s, v=v * scale) for scale in np.linspace(1, 0, min(100, len(x)))])

    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    ax.add_collection(lc)
    if set_ax_lim:
        if not all(np.isnan(x)):
            ax.set_xlim([min(np.nanmin(x), ax.get_xlim()[0]), max(np.nanmax(x), ax.get_xlim()[1])])
        if not all(np.isnan(y)):
            ax.set_ylim([min(np.nanmin(y), ax.get_ylim()[0]), max(np.nanmax(y), ax.get_xlim()[1])])
    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def calc_total_trajectory(df):
    assert 'x' in df.columns and 'y' in df.columns
    return np.sqrt(df.x.diff() ** 2 + df.y.diff() ** 2).sum()


def legend_colors(ax, colors, is_outside=False):
    if is_outside:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    else:
        leg = ax.legend()
    for i, handle in enumerate(leg.legendHandles):
        handle.set_color(colors[i])


def plot_screen(ax):
    rect = patches.Rectangle((200, 1000), 800, 50, linewidth=1, edgecolor='k', facecolor='k')
    ax.add_patch(rect)


def remove_outliers(x, thresh=3, is_replace_nan=False):
    if is_replace_nan:
        idx = np.abs(zscore(x)) >= thresh
        x[idx] = np.nan
    else:
        idx = np.abs(zscore(x)) < thresh
        x = x[idx]
    return x


def put_text(text, frame, x, y, font_scale=1, color=(255, 255, 0), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    :param text: The text to put on frame
    :param frame: The frame numpy array
    :param x: x
    :param y: y
    :param font_scale:
    :param color: default: yellow (255,255,0)
    :param thickness: in px, default 2px
    :param font: font
    :return: frame with text
    """
    return cv2.putText(frame, str(text), (x, y), font, font_scale, color, thickness, cv2.LINE_AA)