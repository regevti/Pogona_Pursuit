import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
#from tqdm.auto import tqdm # py3.7 issues
from tqdm import tqdm
import matplotlib.patches as patches

from Prediction.detector import xyxy_to_centroid, nearest_detection
import Prediction.calibration as calib


def plot_image(detections, img, output_path=None, show_img=True):
    """
    Draw bounding boxes from detections on img and optionally save to file.
    detections - the output detections of a detector
    img - bounding boxes are drawn on this image
    output_path - path to save the image, or None to not save.
    show_img - whether to show the image or not (boolean).
    """

    plt.figure()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)

    if detections is not None:
        # browse detections and draw bounding boxes
        for x1, y1, box_w, box_h, conf in detections:
            color = (0, 0, 1, 1)
            bbox = patches.Rectangle(
                (x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(bbox)
            plt.text(
                x1,
                y1,
                s=str(round(conf, 2)),
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )
    plt.axis("off")

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)

    if show_img:
        plt.show()
    else:
        plt.clf()


def hsv_to_rgb(H, S, V):
    """
    transform angle to BGR color as 3-tuple
    """
    C = S * V
    X = C * (1 - np.abs((H / 60) % 2 - 1))
    m = V - C

    if H >= 0 and H < 60:
        r, g, b = C, X, 0
    elif H >= 60 and H < 120:
        r, g, b = X, C, 0
    elif H >= 120 and H < 180:
        r, g, b = 0, C, X
    elif H >= 180 and H < 240:
        r, g, b = 0, X, C
    elif H >= 240 and H < 300:
        r, g, b = X, 0, C
    else:
        r, g, b = C, 0, X

    def roun(x):
        return int(round(x))

    # return b,g,r
    return roun((b + m) * 255), roun((g + m) * 255), roun((r + m) * 255)


def vec_to_bgr(vec):
    """
    input: 2D vector
    return: 3-tuple, specifying BGR color selected using HSV formula.
    """

    # transform to [-pi,pi] and then to degrees
    angle = np.arctan2(vec[1], vec[0]) * 180 / np.pi

    # transform to [0,360]
    angle = (angle + 360) % 360
    return hsv_to_rgb(angle, 1, 1)


def time_to_bgr(k, arrowWindow):  # DOES NOT WORK - TODO
    # map the relative position of the frame to [0,360] angle and then to hue
    rel = (arrowWindow - k) / arrowWindow
    return hsv_to_rgb(0, 1, rel)


def draw_arrow(
    frame,
    frameCounter,
    centroids,
    arrowWindow,
    k,
    vis_angle=True,
    windowSize=1,
    scale=2.5,
):
    """
    draws the direction of the velocity vector from (arrowWindow) frames back
    directions based on the first discrete derivative of the 2D coordinates of
    windowSize consecutive centroids of the detecions, if both exist
    """

    # initial arrow
    if frameCounter < windowSize:
        return

    # if no prediction at t - windowSize, bo drawing
    if np.isnan(centroids[frameCounter - windowSize, 0]) or np.isnan(
        centroids[frameCounter, 0]
    ):
        return

    arrowBase = tuple(centroids[frameCounter - windowSize].astype(int))
    arrowHead = tuple(centroids[frameCounter].astype(int))

    # scale head for better visibility
    extend_x = scale * (arrowHead[0] - arrowBase[0])
    extend_y = scale * (arrowHead[1] - arrowBase[1])

    new_x = arrowHead[0] + extend_x
    new_y = arrowHead[1] + extend_y

    if new_x < 0:
        new_x = 0
    if new_x > frame.shape[1]:
        new_x = frame.shape[1]

    if new_y < 0:
        new_y = 0
    if new_y > frame.shape[0]:
        new_y = frame.shape[0]

    arrowHead = (new_x, new_y)

    # compute color based on angle or time
    if vis_angle:
        vec_color = vec_to_bgr(
            [arrowHead[0] - arrowBase[0], arrowHead[1] - arrowBase[1]]
        )
    else:
        vec_color = time_to_bgr(k, arrowWindow)

    cv.arrowedLine(
        frame,
        arrowBase,
        arrowHead,
        color=vec_color,
        thickness=2,
        tipLength=0.2,
        line_type=cv.LINE_AA,
    )


def draw_bounding_boxes(frame,
                        detections,
                        color=(0, 0, 255),
                        is_xyxy=True,
                        ):
    """
    frame - a numpy array representing the image.
    detections - [(x, y, x, y, conf)...] bounding boxes array.
    if is_xyxy==False, then assumes detections is xywh
    
    draws bounding boxes on frame (in place).
    """

    font = cv.FONT_HERSHEY_COMPLEX
    scale = 0.4
    thickness = cv.FILLED
    margin = 4

    if detections is not None:
        for x1, y1, c, d, conf in detections:
            x1 = int(x1)
            y1 = int(y1)

            x2, y2, box_w, box_h = None, None, None, None
            if is_xyxy:
                x2 = int(c)
                y2 = int(d)
            else:
                box_w = int(c)
                box_h = int(d)

            text = str(round(conf, 2))
            txt_size = cv.getTextSize(text, font, scale, thickness)
            end_x = int(x1 + txt_size[0][0] + margin)
            end_y = int(y1 + txt_size[0][1] + margin)

            cv.rectangle(frame, (x1, y1), (end_x, end_y), color, thickness)

            if is_xyxy:
                cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                cv.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 2)
            cv.putText(
                frame,
                text,
                (x1, end_y - margin),
                font,
                scale,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )


def draw_k_arrows(
    frame, frameCounter, centroids, arrowWindow, visAngle, windowSize, scale=5
):
    for k in range(arrowWindow):
        draw_arrow(
            frame,
            frameCounter - k,
            centroids,
            arrowWindow,
            k,
            visAngle,
            windowSize,
            scale,
        )


def draw_k_centroids(frame, frameCounter, centroids, k, color=(0, 0, 255)):
    if k > frameCounter:
        k = frameCounter - 1

    for j in range(k):
        if np.isnan(centroids[frameCounter - j][0]):
            continue
        x = int(centroids[frameCounter - j][0])
        y = int(centroids[frameCounter - j][1])
        cv.circle(
            frame,
            center=(x, y),
            radius=2,
            color=color,
            thickness=-1,
            lineType=cv.LINE_AA,
        )


def online_centroid_visualizer(detector, color, window_size):
    centroids = []

    def fn(orig_frame, write_frame, width, height, frame_counter):
        detector.set_input_size(width, height)
        detections = detector.detect_image(orig_frame)
        if detections is not None:
            if len(centroids) > 0:
                detection = nearest_detection(detections, centroids[-1])
            else:
                detection = detections[0]

            centroid = xyxy_to_centroid(detection)
            if len(centroids) >= window_size:
                centroids.pop(0)

            centroids.append(centroid)

        for c in centroids:
            if np.isnan(c[0]):
                continue

            x = int(c[0])
            y = int(c[1])
            cv.circle(
                write_frame,
                center=(x, y),
                radius=2,
                color=color,
                thickness=-1,
                lineType=cv.LINE_AA,
            )

    return fn


def missed_frames_saver(
    detector, output_dir, prefix="frame", save_thresh=0.8, above=False, draw_bbox=True
):
    saved_counter = 0

    if above:

        def save_func(detec, save_thres):
            return (detec is not None) and (detec[0][4] > save_thres)

    else:

        def save_func(detec, save_thres):
            return (detec is None) or (detec[0][4] < save_thres)

    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass

    def fn(orig_frame, write_frame, frame_counter):
        nonlocal saved_counter

        detections = detector.detect_image(orig_frame)

        if save_func(detections, save_thresh):
            if detections is not None:
                prob = str(round(detections[0][4], 3))
            else:
                prob = "0"

            save_path = os.path.join(
                output_dir, prefix + "_" + prob + "_" + str(saved_counter) + ".jpg"
            )

            if draw_bbox:
                plot_image(
                    detections, orig_frame, output_path=save_path, show_plot=False
                )
            else:
                cv.imwrite(save_path, orig_frame)

            saved_counter += 1

    return fn


def video_sampler(output_path, freq, f_name):
    """
    Save arbitrary frames at constant frequency from video to path
    """

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    def fn(orig_frame, write_frame, frame_counter):
        if frame_counter % freq == 0:
            fname = os.path.join(output_path, f_name + str(frame_counter) + ".jpg")
            cv.imwrite(fname, orig_frame)

    return fn


def offline_centroid_visualizer(centroids, color, window_size):
    def fn(orig_frame, write_frame, frame_counter):
        draw_k_centroids(write_frame, frame_counter, centroids, window_size, color)

    return fn


def offline_bbox_visualizer(bboxes, color=(0, 0, 255), window_size=20):
    def fn(orig_frame, write_frame, frame_counter):
        for i in range(window_size):
            idx = max(frame_counter - i, 0)
            bbox = bboxes[idx]
            if not np.isnan(bbox[0]):
                bbox = bbox * (bbox > 0)  # zero out negative coords.
                bbox = bbox.astype(int)
                cv.rectangle(write_frame, tuple(bbox[:2]), tuple(bbox[2:]), color, 1)

    return fn


def offline_arrow_visualizer(centroids, window_size, vis_angle=True, scale=5):
    def fn(orig_frame, write_frame, frame_counter):
        draw_k_arrows(
            write_frame, frame_counter, centroids, window_size, vis_angle, 2, scale
        )

    return fn


def offline_kalman_visualizer(cents_df, max_k):
    def fn(orig_frame, write_frame, frame_counter):
        if any(np.isnan(cents_df[["det_x", "det_y"]].iloc[frame_counter])):
            return

        orig = tuple(
            cents_df[["det_x", "det_y"]].iloc[frame_counter].values.astype(int)
        )
        pred = tuple(
            cents_df[["pred_x", "pred_y"]].iloc[frame_counter].values.astype(int)
        )

        k = int(cents_df.k[frame_counter])

        cv.circle(write_frame, orig, radius=2, color=(0, 0, 255), thickness=-1)
        if k == max_k - 1:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv.line(write_frame, orig, pred, color=color, thickness=2, lineType=cv.LINE_AA)

    return fn


def is_point_in_bounds(p, frame):
    px, py = p
    return px >= 0 and px < frame.shape[1] and py >= 0 and py < frame.shape[0]


def visualize_prediction(
    predictor, write_frame, forecast, hit_x, hit_steps, show_forecast_bbox=False
):
    bbox = predictor.history[predictor.frame_num - 1]
    if not np.isnan(bbox[0]):
        c = xyxy_to_centroid(bbox).astype(int)
        if is_point_in_bounds(c, write_frame):
            cv.circle(
                write_frame,
                center=tuple(c),
                radius=4,
                color=(255, 0, 0),
                thickness=-1,
                lineType=cv.LINE_AA,
            )

        bbox = bbox * (bbox > 0)  # zero out negative coords.
        bbox = bbox.astype(int)
        cv.rectangle(write_frame, tuple(bbox[:2]), tuple(bbox[2:]), (255, 0, 0), 1)

    if forecast is not None:
        for bbox in forecast:
            bbox = bbox.astype(int)
            bbox = bbox * (bbox > 0)  # zero out negative coords.

            # center of bottom bbox edge
            c = xyxy_to_centroid(bbox).astype(int)
            if is_point_in_bounds(c, write_frame):
                cv.circle(
                    write_frame,
                    center=(c[0], bbox[3]),
                    radius=2,
                    color=(0, 255, 0),
                    thickness=-1,
                    lineType=cv.LINE_AA,
                )

            if show_forecast_bbox:

                cv.rectangle(
                    write_frame, tuple(bbox[:2]), tuple(bbox[2:]), (255, 255, 0), 1
                )


    if hit_x is not None:
        hp = (int(hit_x), predictor.prediction_y_threshold)

        if is_point_in_bounds(hp, write_frame):
            cv.circle(
                write_frame,
                center=hp,
                radius=4,
                color=(0, 0, 255),
                thickness=-1,
                lineType=cv.LINE_AA,
            )

            cv.putText(
                write_frame,
                str(hit_steps),
                (hp[0], hp[1] - 20),
                cv.FONT_HERSHEY_COMPLEX,
                fontScale=0.4,
                color=(255, 0, 0),
                thickness=1,
                lineType=cv.LINE_AA,
            )


def offline_predictor_visualizer(predictor, bboxes, show_forecast_bbox=False):
    def fn(orig_frame, write_frame, frame_counter):
        forecast, hit_point, hit_steps = predictor.handle_detection(
            bboxes[frame_counter]
        )
        visualize_prediction(
            predictor,
            write_frame,
            forecast,
            hit_point,
            hit_steps,
            show_forecast_bbox=show_forecast_bbox,
        )

    return fn


def predictor_visualizer(predictor, show_forecast_bbox=False):
    def fn(orig_frame, write_frame, frame_counter):
        forecast, hit_point, hit_steps = predictor.handle_frame(orig_frame)
        visualize_prediction(
            predictor, write_frame, forecast, hit_point, hit_steps, show_forecast_bbox
        )

    return fn


def get_correction_fn(
    homography, screen_x_res, mtx=calib.MTX, dist=calib.DIST  # 1920
):
    """
    Receive the parameters to conduct the lense correction and coordinate transformation
    on the write_frame image.

    :param aff_mat: affine transformation to touch screen coordinates
    :param screen_x_res: screen resolution
    :param mtx: camera model matrix
    :param dist: distortion matrix
    :return:
    """

    first_frame = True
    #  compute the distortion mapping once, according to (w, h)
    mapx, mapy, roi = None, None, None

    def fn(frame):
        nonlocal mapx, mapy, roi, first_frame
        if first_frame:
            first_frame = False
            frame_height, frame_width = frame.shape[0], frame.shape[1]
            (mapx, mapy), roi, _ = calib.get_undistort_mapping(
                frame_width, frame_height, mtx=mtx, dist=dist
            )

        # write over write frame and use the corrected
        write_frame = calib.undistort_image(frame, (mapx, mapy))
        if homography is not None:
            write_frame = calib.transform_image(write_frame, homography, screen_x_res)
        return write_frame

    return fn


def process_video(
    video_path,
    output_path,
    process_fns,
    correction_fn=None,
    start_frame=0,
    num_frames=None,
    frame_rate=None,
    resize_to_width=None,
):
    """
    Open a video file, run all functions in process_fns, and write the
    processed video to file.

    video_path - path of original video
    output_path - path of processed video or None if writing is not necessary.
    process_fns - a list of process functions
    correction_fn - a function responsible for undistorting and transforming the image
    start_frame - the first frame to be processed.
    num_frames - number of frames to process or None to process the whole video.
    frame_rate - the framerate of the processed video or None to use the original framerate.
    resize_to_width - when not None, the output is resized after processing each frame.

    Each function in process_fns has the following signature:

    fn(orig_frame, write_frame, frame_counter)

    orig_frame - the original video frame.
    write_frame - the frame that will be written to file.
    frame_counter - current frame number.

    correction_fn has the following signature:
    fn(input_frame) -> output_frame
    output_frame is the transformed frame.
    """

    vcap = cv.VideoCapture(video_path)

    if start_frame != 0:
        vcap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    if num_frames is None:
        num_frames = int(vcap.get(cv.CAP_PROP_FRAME_COUNT)) - start_frame

    if frame_rate is None:
        frame_rate = vcap.get(cv.CAP_PROP_FPS)

    for frame_counter in tqdm(range(start_frame, start_frame + num_frames)):
        ret, orig_frame = vcap.read()

        if not ret:
            print("error reading frame")
            break

        if correction_fn is not None:
            orig_frame = correction_fn(orig_frame)

        write_frame = orig_frame.copy()

        if output_path is not None and frame_counter == start_frame:
            if resize_to_width is not None:
                rwidth = resize_to_width
                rheight = int(
                    write_frame.shape[0] * resize_to_width / write_frame.shape[1]
                )
            else:
                rwidth = write_frame.shape[1]
                rheight = write_frame.shape[0]

            videowriter = cv.VideoWriter(
                output_path,
                cv.VideoWriter_fourcc(*"mp4v"),
                frame_rate,
                (rwidth, rheight),
            )

        for fn in process_fns:
            fn(orig_frame, write_frame, frame_counter)

        if output_path is not None:
            if resize_to_width is not None:
                write_frame = cv.resize(write_frame, (rwidth, rheight))

            videowriter.write(write_frame)

    vcap.release()

    if output_path is not None:
        videowriter.release()
