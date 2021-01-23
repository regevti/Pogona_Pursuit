import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll


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


def colorline(ax, x, y, z=None, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
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
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    ax.add_collection(lc)

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